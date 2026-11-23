from policy import looks_like_pii, is_advice_or_performance


def test_pan_is_rejected():
    assert looks_like_pii("My PAN is ABCDE1234F")


def test_email_is_rejected():
    assert looks_like_pii("Contact me at person@example.com")


def test_phone_is_rejected():
    assert looks_like_pii("My number is 9876543210")


def test_aadhaar_like_number_is_rejected():
    assert looks_like_pii("Aadhaar 123456789012")


def test_buy_advice_is_refused():
    assert is_advice_or_performance("Should I buy HDFC Flexi Cap Fund?")


def test_return_comparison_is_refused():
    assert is_advice_or_performance("Compare returns of these funds")


def test_factual_question_is_not_advice():
    assert not is_advice_or_performance("What is the exit load?")
