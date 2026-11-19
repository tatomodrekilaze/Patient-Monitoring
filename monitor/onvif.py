"""get the rtsp url from an ip camera over onvif.

some cams (conico / MIPC firmware) give out a new rtsp url with a random code every time and the
old one stops working, so we resolve onvif://<ip> (main) or onvif://<ip>/sub on every reconnect.
"""
import re
import time
import urllib.request

PORTS = (80, 8899, 8000, 8080, 2000, 5000, 6688, 10080)
PATHS = ("onvif/media_service", "onvif/device_service", "onvif/Media")
SOAP = ('<?xml version="1.0" encoding="UTF-8"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
        'xmlns:trt="http://www.onvif.org/ver10/media/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">'
        '<s:Body>{}</s:Body></s:Envelope>')
STREAM_URI = ('<trt:GetStreamUri><trt:StreamSetup><tt:Stream>RTP-Unicast</tt:Stream><tt:Transport>'
              '<tt:Protocol>RTSP</tt:Protocol></tt:Transport></trt:StreamSetup>'
              '<trt:ProfileToken>{}</trt:ProfileToken></trt:GetStreamUri>')

_endpoint_cache = {}
_info_cache = {}


def _soap(url, body, timeout=4):
    req = urllib.request.Request(url, SOAP.format(body).encode(),
                                 {"Content-Type": "application/soap+xml; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "ignore")
    except Exception:
        return ""


def streams(host):
    """[(profile token, rtsp uri)] - main profile first. Empty if the camera does not answer."""
    candidates = [_endpoint_cache[host]] if host in _endpoint_cache else \
        [f"http://{host}:{p}/{path}" for p in PORTS for path in PATHS]
    for url in candidates:
        tokens = re.findall(r'<(?:\w+:)?Profiles[^>]*token="([^"]+)"', _soap(url, "<trt:GetProfiles/>"))
        out = []
        for tok in tokens:
            uri = re.findall(r"<(?:\w+:)?Uri>([^<]+)<", _soap(url, STREAM_URI.format(tok)))
            if uri:
                out.append((tok, uri[0].replace("&amp;", "&")))
        if out:
            _endpoint_cache[host] = url
            return out
    return []


def host_of(source):
    """Camera host from onvif://host[/sub] or rtsp://[user:pw@]host[:port]/..., else None."""
    m = re.match(r"^(?:onvif|rtsp|http|https)://(?:[^/@]*@)?([^/:?]+)", source)
    return m.group(1) if m else None


def _device_url(host):
    if host in _endpoint_cache:
        return _endpoint_cache[host].rsplit("/onvif/", 1)[0] + "/onvif/device_service"
    return f"http://{host}/onvif/device_service"


def device_info(host):
    """Manufacturer, model, firmware - or {} when the camera does not answer."""
    xml = _soap(_device_url(host), '<tds:GetDeviceInformation xmlns:tds="http://www.onvif.org/ver10/device/wsdl"/>')
    out = {}
    for k in ("Manufacturer", "Model", "FirmwareVersion", "SerialNumber"):
        m = re.search(rf"<(?:\w+:)?{k}>([^<]*)<", xml)
        if m:
            out[k.lower()] = m.group(1).strip()
    return out


def cached_info(host):
    """Device info and profiles saved when the stream was opened ({} if never connected)."""
    return _info_cache.get(host, {})


def profiles(host, query=False):
    """[{token, name, width, height, fps, codec}] for the stream picker. Without query=True only the
    copy saved at connect time is returned (asking a streaming Conico kills its stream)."""
    if not query:
        return _info_cache.get(host, {}).get("profiles", [])
    if host not in _endpoint_cache and not streams(host):
        return []
    xml = _soap(_endpoint_cache[host], "<trt:GetProfiles/>")
    out = []
    for block in re.findall(r"<(?:\w+:)?Profiles\b.*?</(?:\w+:)?Profiles>", xml, re.S):
        tok = re.search(r'token="([^"]+)"', block)
        enc = re.search(r"<(?:\w+:)?VideoEncoderConfiguration\b.*?</(?:\w+:)?VideoEncoderConfiguration>", block, re.S)
        e = enc.group(0) if enc else ""
        num = lambda tag: (re.search(rf"<(?:\w+:)?{tag}>(\d+)<", e) or [None, None])[1]   # noqa: E731
        name = re.search(r"<(?:\w+:)?Name>([^<]+)<", block)
        codec = re.search(r"<(?:\w+:)?Encoding>([^<]+)<", e)
        out.append({"token": tok.group(1) if tok else None, "name": name.group(1) if name else "",
                    "width": int(num("Width") or 0), "height": int(num("Height") or 0),
                    "fps": int(num("FrameRateLimit") or 0), "codec": codec.group(1) if codec else ""})
    return out


def _ptz_url(host):
    return _device_url(host).replace("/onvif/device_service", "/onvif/ptz")


PTZ = ('<?xml version="1.0" encoding="UTF-8"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
       'xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">'
       '<s:Body>{}</s:Body></s:Envelope>')


def _ptz(host, body):
    req = urllib.request.Request(_ptz_url(host), PTZ.format(body).encode(),
                                 {"Content-Type": "application/soap+xml; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def ptz_move(host, pan=0.0, tilt=0.0, zoom=0.0, token="MainProfileToken"):
    """Start moving at the given speeds (-1..1). The Conico FI-362C accepts ContinuousMove / Stop
    without a login (it does not answer the read-only PTZ calls, so support is found by trying)."""
    zoom_xml = f'<tt:Zoom x="{zoom:.2f}"/>' if zoom else ""
    return _ptz(host, f'<tptz:ContinuousMove><tptz:ProfileToken>{token}</tptz:ProfileToken><tptz:Velocity>'
                      f'<tt:PanTilt x="{pan:.2f}" y="{tilt:.2f}"/>{zoom_xml}</tptz:Velocity></tptz:ContinuousMove>')


def ptz_stop(host, token="MainProfileToken"):
    return _ptz(host, f"<tptz:Stop><tptz:ProfileToken>{token}</tptz:ProfileToken>"
                      "<tptz:PanTilt>true</tptz:PanTilt><tptz:Zoom>true</tptz:Zoom></tptz:Stop>")


def reachable(host, timeout=1.5):
    """Does anything answer on the camera's ONVIF ports? Asked first, so a camera that is off or on
    another network fails in a second instead of minutes of SOAP time-outs."""
    import socket
    from concurrent.futures import ThreadPoolExecutor

    def knock(p):
        try:
            with socket.create_connection((host, p), timeout=timeout):
                return True
        except OSError:
            return False
    with ThreadPoolExecutor(len(PORTS)) as ex:          # all ports at once: ~1.5 s, not 8 x 1.5 s
        return any(ex.map(knock, PORTS))


def resolve(source):
    """onvif://host[/sub] -> current rtsp:// address (or the source unchanged if it is not onvif://)."""
    if not source.startswith("onvif://"):
        return source
    rest = source[len("onvif://"):]
    host, _, which = rest.partition("/")
    found = []
    if not reachable(host):
        raise RuntimeError(f"ONVIF camera {host} is not reachable")
    for attempt in range(3):            # right after a stream closes the Conico ignores ONVIF for ~1 s
        found = streams(host)
        if found:
            break
        time.sleep(1.0)
    # conico drops the stream on ANY onvif request while streaming, so grab everything now and cache it
    if found and host not in _info_cache:
        _info_cache[host] = {"device": device_info(host), "profiles": profiles(host, query=True)}
    if not found:
        raise RuntimeError(f"ONVIF camera {host} did not report a stream")
    # default = the full HD main stream: pose runs on a 640 px copy, but the chest / hands / face are
    # cut from the full picture (breathing in a hard synthetic test: 38-75 % -> 100 % within 2/min).
    # The smearing it once showed came from ONVIF queries killing the stream, now avoided; if Wi-Fi
    # still cannot carry it, Capture falls back to /sub by itself. /sub asks for the light stream.
    if which.lower().startswith("sub") and len(found) > 1:
        return found[1][1]
    return found[0][1]
