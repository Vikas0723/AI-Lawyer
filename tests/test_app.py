from modules.assistant import is_legal_query

def test_general_question():
    assert not is_legal_query("What is an IP address?")

def test_legal_question():
    assert is_legal_query("My landlord is not returning my security deposit")

def test_hindi_legal_question():
    assert is_legal_query("मकान मालिक मेरा किराया जमा वापस नहीं कर रहा है")
