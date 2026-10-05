import uvicorn
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    print(f"\n==================================================")
    print(f"  Dry Eye Detection AI Backend Server")
    print(f"  Hybrid System: 80% Thermal CNN + 20% OSDI XGBoost")
    print(f"  Listening on: http://0.0.0.0:{port}")
    print(f"  Interactive API Docs: http://localhost:{port}/docs")
    print(f"==================================================\n")
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=False)
