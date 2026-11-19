"""Find the video stream address of a Wi-Fi security camera.

    python tools/find_camera_stream.py 192.168.50.43
    python tools/find_camera_stream.py 192.168.50.43 --user admin

1. Asks the camera itself over ONVIF (the standard most IP cameras speak): GetProfiles + GetStreamUri.
   Many cheap cameras answer without a login - the Conico FI-362C (MIPC firmware) does, and its
   address (rtsp://IP:554/live/<id>_p0_<code>) is not one anybody could guess.
2. Only if that fails: asks for the camera password (hidden typing, never printed or saved) and tries
   the RTSP / HTTP paths common on cheap cameras.
Prints the run.py command for the stream that gives a picture.
"""
import argparse
import getpass
import os
import re
import sys
import time
import urllib.request
from urllib.parse import quote

import cv2

ONVIF_PORTS = (80, 8899, 8000, 8080, 2000, 5000, 6688, 10080)
RTSP_PATHS = ["stream0", "stream1", "", "live", "live/ch0", "live/ch00_0", "11", "12", "ch0", "onvif1",
              "h264", "main", "media/video1", "user={user}&password={password}&channel=1&stream=0.sdp",
              "cam/realmonitor?channel=1&subtype=0", "Streaming/Channels/101"]
HTTP_PATHS = ["video.mjpg", "videostream.cgi?user={user}&pwd={password}", "mjpeg.cgi", "video", "stream"]

SOAP = ('<?xml version="1.0" encoding="UTF-8"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
        'xmlns:trt="http://www.onvif.org/ver10/media/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">'
        '<s:Body>{}</s:Body></s:Envelope>')
STREAM_URI = ('<trt:GetStreamUri><trt:StreamSetup><tt:Stream>RTP-Unicast</tt:Stream><tt:Transport>'
              '<tt:Protocol>RTSP</tt:Protocol></tt:Transport></trt:StreamSetup>'
              '<trt:ProfileToken>{}</trt:ProfileToken></trt:GetStreamUri>')


def soap(url, body):
    req = urllib.request.Request(url, SOAP.format(body).encode(),
                                 {"Content-Type": "application/soap+xml; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=4) as r:
            return r.read().decode("utf-8", "ignore")
    except Exception:
        return ""


def onvif_streams(ip):
    """[(profile, rtsp uri)] from the camera's ONVIF media service, without a login."""
    for port in ONVIF_PORTS:
        for path in ("onvif/media_service", "onvif/device_service", "onvif/Media"):
            url = f"http://{ip}:{port}/{path}"
            xml = soap(url, "<trt:GetProfiles/>")
            tokens = re.findall(r'<(?:\w+:)?Profiles[^>]*token="([^"]+)"', xml)
            if not tokens:
                continue
            out = []
            for tok in tokens:
                uri = re.findall(r"<(?:\w+:)?Uri>([^<]+)<", soap(url, STREAM_URI.format(tok)))
                if uri:
                    out.append((tok, uri[0].replace("&amp;", "&")))
            if out:
                print(f"ONVIF answered on port {port} with {len(out)} stream(s)")
                return out
    return []


def try_open(url, seconds=6):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame.shape
            if not cap.isOpened():
                return None
            time.sleep(0.2)
    finally:
        cap.release()
    return None


def found(source, shape):
    print(f"WORKS ({shape[1]}x{shape[0]})")
    print("\nStart the monitor with this camera:\n")
    print(f'  python run.py --source "{source}"\n')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ip")
    ap.add_argument("--user", default="admin")
    a = ap.parse_args()

    streams = onvif_streams(a.ip)
    for tok, uri in streams:
        print(f"trying ONVIF {tok}: {re.sub(r'//[^/@]*@', '//***@', uri)} ...", end=" ", flush=True)
        shape = try_open(uri)
        if shape:
            print(f"WORKS ({shape[1]}x{shape[0]})")
            sub = "/sub" if tok != streams[0][0] else ""
            print("\nThis camera gives out a new address every time (old ones expire), so let the monitor")
            print("ask it over ONVIF on each start / reconnect:\n")
            print(f'  python run.py --source "onvif://{a.ip}{sub}"\n')
            return 0
        print("needs a login" if "@" not in uri else "no")

    pw = getpass.getpass(f"Password for {a.user}@{a.ip} (hidden): ")
    q = quote(pw, safe="")
    cands = [f"rtsp://{{user}}:{{password}}@{a.ip}:554/{p}" for p in RTSP_PATHS]
    cands += [f"http://{{user}}:{{password}}@{a.ip}/{p}" for p in HTTP_PATHS]
    for tmpl in cands:
        shown = tmpl.replace("{user}", a.user)                 # the password placeholder stays as text
        print(f"trying {shown.replace('{password}', '***')} ...", end=" ", flush=True)
        shape = try_open(shown.replace("{password}", q))
        if shape:
            print("(run.py will ask for the password)")
            return found(shown, shape)
        print("no")
    print("\nNo stream found. Look in the camera's app for RTSP / ONVIF and switch it on, then run this again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
