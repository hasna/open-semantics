# SemHex

Semantic Hexadecimal Encoding — a universal compact discrete encoding for meaning, like hex codes for colors.

```bash
pip install semhex
```

```python
import semhex

codes = semhex.encode("Can you help me debug this async error?")
# → ["$3A.C8F0", "$72.B1A0"]

decoded = semhex.decode(codes)
# → "Request for help with code (async timeout)"

dist = semhex.distance("$8A.2100", "$8A.2400")
# → 0.15 (anger ↔ frustration: close)

blended = semhex.blend("$8A.2100", "$60.3000")
# → "$8A.0900" (anger + small = annoyance)
```
