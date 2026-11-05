import struct


LOGF_SOURCE = "https://codebrowser.dev/glibc/glibc/sysdeps/ieee754/flt-32/e_logf.c.html"
LOGF_OFFSET = 0x3F330000
LOGF_LN2 = float.fromhex("0x1.62e42fefa39efp-1")
LOGF_POLY = tuple(float.fromhex(x) for x in (
    "-0x1.00ea348b88334p-2",
    "0x1.5575b0be00b6ap-2",
    "-0x1.ffffef20a4123p-2",
))
LOGF_TABLE = tuple((float.fromhex(a), float.fromhex(b)) for a, b in (
    ("0x1.661ec79f8f3bep+0", "-0x1.57bf7808caadep-2"),
    ("0x1.571ed4aaf883dp+0", "-0x1.2bef0a7c06ddbp-2"),
    ("0x1.49539f0f010bp+0", "-0x1.01eae7f513a67p-2"),
    ("0x1.3c995b0b80385p+0", "-0x1.b31d8a68224e9p-3"),
    ("0x1.30d190c8864a5p+0", "-0x1.6574f0ac07758p-3"),
    ("0x1.25e227b0b8eap+0", "-0x1.1aa2bc79c81p-3"),
    ("0x1.1bb4a4a1a343fp+0", "-0x1.a4e76ce8c0e5ep-4"),
    ("0x1.12358f08ae5bap+0", "-0x1.1973c5a611cccp-4"),
    ("0x1.0953f419900a7p+0", "-0x1.252f438e10c1ep-5"),
    ("0x1p+0", "0x0p+0"),
    ("0x1.e608cfd9a47acp-1", "0x1.aa5aa5df25984p-5"),
    ("0x1.ca4b31f026aap-1", "0x1.c5e53aa362eb4p-4"),
    ("0x1.b2036576afce6p-1", "0x1.526e57720db08p-3"),
    ("0x1.9c2d163a1aa2dp-1", "0x1.bc2860d22477p-3"),
    ("0x1.886e6037841edp-1", "0x1.1058bc8a07ee1p-2"),
    ("0x1.767dcf5534862p-1", "0x1.4043057b6ee09p-2"),
))
LOG10_SPLIT_LO, LOG10_SPLIT_IV, LOG10_SPLIT_HI = (
    struct.unpack("!f", struct.pack("!I", bits))[0]
    for bits in (0x355427DB, 0x3EDE5BD9, 0x3E9A2080)
)
