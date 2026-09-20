"""CAD generation: printable geometry, printability checks and print sheets."""

from cmtool.cad.coupons import (
    CantileverCouponSpec,
    CouponSet,
    FlexureCouponSpec,
    build_cantilever_coupon,
    build_flexure_coupon,
)
from cmtool.cad.export import (
    PrintabilityCheck,
    bounding_box_mm,
    check_printability,
    export_solid,
    write_print_sheet,
)

__all__ = [
    "CantileverCouponSpec",
    "CouponSet",
    "FlexureCouponSpec",
    "PrintabilityCheck",
    "bounding_box_mm",
    "build_cantilever_coupon",
    "build_flexure_coupon",
    "check_printability",
    "export_solid",
    "write_print_sheet",
]
