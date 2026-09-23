from .montgomery_curve import AffMontgomeryPoint, ECMontgomery


def test_ECMontgomery() -> None:

    ec = ECMontgomery(17, 3, 1)
    p1 = AffMontgomeryPoint(0, 0, ec)
    p2 = AffMontgomeryPoint.infinity(ec)
    assert p1 + p2 == AffMontgomeryPoint(0, 0, ec)

    p3 = AffMontgomeryPoint.infinity(ec)
    assert p2 + p3 == p3

    p1 = AffMontgomeryPoint.random(ec)
    print(p1 * 10)  # doing this to test if the formulas look good
