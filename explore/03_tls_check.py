"""Phase 0: diagnose TLS failure of aiohttp against servapibi.xm.com.co."""

import socket
import ssl

import certifi

HOST = "servapibi.xm.com.co"


def try_ctx(name: str, ctx: ssl.SSLContext) -> None:
    try:
        with (
            socket.create_connection((HOST, 443), timeout=20) as s,
            ctx.wrap_socket(s, server_hostname=HOST) as ss,
        ):
            cert = ss.getpeercert()
            print(f"[OK]   {name}: issuer={dict(x[0] for x in cert['issuer'])}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {e}")


print("openssl:", ssl.OPENSSL_VERSION)
print("default verify paths:", ssl.get_default_verify_paths())
try_ctx("ssl.create_default_context()", ssl.create_default_context())
try_ctx("certifi bundle", ssl.create_default_context(cafile=certifi.where()))
