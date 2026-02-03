# Binance Demo Account & API Configuration Guide

## 🔄 Demo vs Testnet vs Production

**In this project:**

| Mode | `BINANCE_TESTNET` | Endpoint | Use case |
|------|-------------------|----------|----------|
| **Demo Trading** (recommended) | `true` | `https://demo-fapi.binance.com` | Virtual funds, full features |
| **Production / Real** | `false` | `https://fapi.binance.com` | Real money |

Demo API keys from Binance Futures → Demo Trading use **demo-fapi.binance.com**. Set `testnet=True` in code and `BINANCE_TESTNET=true` in `.env`.

---

## 📝 Step-by-Step: Getting Demo API Keys

### Option 1: Binance Futures Demo Trading (Recommended)

1. **Login to Binance**  
   Visit https://www.binance.com/ and log in.

2. **Switch to Demo Trading**  
   Go to **Derivatives** → **USDT-M Futures** → toggle **Demo Trading** (top-right).

3. **Create API keys in Demo mode**  
   While in Demo: **Profile** → **API Management** (or https://www.binance.com/en/my/settings/api-management) → **Create API**.

4. **Permissions**  
   - ✅ Enable Futures  
   - ✅ Enable Reading  
   - ❌ Do not enable Withdrawal  

5. **Save keys**  
   Copy API Key and Secret Key (Secret is shown only once).

6. **Configure `.env`** (in `modules/auto_trade/`):
   ```bash
   BINANCE_API_KEY=your_demo_api_key_here
   BINANCE_API_SECRET=your_demo_secret_here
   BINANCE_TESTNET=true
   ```

7. **In code** use `testnet=True` (e.g. in `test_demo_api.py`):
   ```python
   client = BinanceClient(
       api_key=os.getenv("BINANCE_API_KEY"),
       api_secret=os.getenv("BINANCE_API_SECRET"),
       testnet=True,
   )
   ```

### Option 2: Old Testnet (Fallback – may be deprecated)

- Visit https://testnet.binancefuture.com/, log in with GitHub, generate keys.
- Set `BINANCE_TESTNET=true`.  
- Note: Current `binance_client.py` routes `testnet=True` to **demo-fapi.binance.com**, not testnet.binancefuture.com. For old testnet you would need a separate URL config.

---

## 🔧 Technical Note: Endpoint Override (Demo Fix)

Balance/position calls use CCXT’s **fapiPrivateV2**, not only `public`/`private`. Overriding only `public` and `private` left fapiPrivateV2 pointing at production and caused auth errors.

**Fix in `binance_client.py`** (when `testnet=True`): override all futures URLs:

```python
config["urls"] = {
    "api": {
        "fapiPublic": "https://demo-fapi.binance.com/fapi/v1",
        "fapiPublicV2": "https://demo-fapi.binance.com/fapi/v2",
        "fapiPublicV3": "https://demo-fapi.binance.com/fapi/v3",
        "fapiPrivate": "https://demo-fapi.binance.com/fapi/v1",
        "fapiPrivateV2": "https://demo-fapi.binance.com/fapi/v2",
        "fapiPrivateV3": "https://demo-fapi.binance.com/fapi/v3",
        "fapiData": "https://demo-fapi.binance.com/futures/data",
    }
}
```

**Verification in `test_demo_api.py`:** the script prints the endpoints in use (e.g. `fapiPrivateV2`, `fapiPrivate`) so you can confirm they point to `demo-fapi.binance.com` when using demo keys.

---

## ✅ Verify Your Setup

1. **Check `.env`**  
   `BINANCE_TESTNET=true` for demo; keys from Demo Trading.

2. **Run test**
   ```bash
   python modules/auto_trade/test_demo_api.py
   ```

3. **Expected output (success)**  
   - "Initialized Binance Demo client (uses demo-fapi.binance.com)"  
   - fapiPrivateV2: `https://demo-fapi.binance.com/fapi/v2`  
   - Balance check and positions check OK (if keys are valid).

---

## 🆘 Troubleshooting

| Error | Cause | Fix |
|-------|--------|-----|
| **-2008 Invalid Api-Key ID** | Expired or wrong key | Create new keys in Demo Trading, update `.env`. |
| **-2015 Invalid API-key, IP, or permissions** | IP restriction or missing Futures permission | Enable Futures, set IP to Unrestricted for testing. |
| **-1022 Signature invalid** | Wrong secret or time sync | Check secret in `.env`, sync system time. |
| Balance/positions fail but endpoint looks right | Keys not from Demo mode | Create keys while in **Demo Trading** (not main account). |

**Rule of thumb:**  
- **Demo keys** → `BINANCE_TESTNET=true` → demo-fapi.binance.com  
- **Real keys** → `BINANCE_TESTNET=false` → fapi.binance.com  

---

## 📊 Demo Account Features

- Virtual funds (e.g. 10,000 USDT).  
- Full futures features: orders, TP/SL, strategies.  
- No real money at risk.  
- Balance can be reset in Binance Demo.

---

## 📚 Links

- Binance Futures API: https://binance-docs.github.io/apidocs/futures/en/
- Demo / Testnet FAQ: https://www.binance.com/en/support/faq/how-to-test-my-functions-on-binance-futures-testnet-ab78f9a1b8824cf0a106b4229c76496d
- CCXT Binance: https://docs.ccxt.com/#/exchanges/binance

---

## 🔐 Security

- Demo/Testnet keys: safe for testing.  
- Real keys: never commit, never share, no withdrawal permission, use IP whitelist and 2FA.

---

**Last updated:** 2026-02  
**Status:** Demo = `testnet=True` → demo-fapi.binance.com; Production = `testnet=False` → fapi.binance.com
