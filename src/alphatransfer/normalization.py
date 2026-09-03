"""Explicit direction and nominal transformations."""


def cbr_rub_per_unit(value: float, nominal: float) -> float:
    if value <= 0 or nominal <= 0:
        raise ValueError("CBR value and nominal must be positive")
    return value / nominal


def nbk_rub_per_kzt(kzt_value: float, nominal: float) -> float:
    if kzt_value <= 0 or nominal <= 0:
        raise ValueError("NBK value and nominal must be positive")
    return nominal / kzt_value


def moex_rub_per_kzt(price: float, facevalue: float) -> float:
    if price <= 0 or facevalue <= 0:
        raise ValueError("MOEX price and FACEVALUE must be positive")
    return price / facevalue


def cross_rub_per_kzt(cbr_rub_per_fx: float, nbk_kzt_per_fx: float) -> float:
    if cbr_rub_per_fx <= 0 or nbk_kzt_per_fx <= 0:
        raise ValueError("cross inputs must be positive")
    return cbr_rub_per_fx / nbk_kzt_per_fx
