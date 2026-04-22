"""Project-wide constants.

Bund proxy choice — IS0L.DE (iShares Germany Govt Bond UCITS ETF, EUR Dist).
Source: https://finance.yahoo.com/quote/IS0L.DE/
Rationale: tradeable instrument with continuous EUR pricing on Yahoo Finance,
unlike yield series (^TNX-style) which are not directly available for Bund.
The ETF tracks the Bloomberg Germany Treasury index → acceptable proxy for
10y Bund *price* dynamics (the project asks for prices, not yields).
"""

TICKERS = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Oil": "CL=F",
    "EURUSD": "EURUSD=X",
    "JPYUSD": "JPYUSD=X",
    "DXY": "DX-Y.NYB",
    "UST10Y": "^TNX",
    "Bund10Y": "IS0L.DE",
}
