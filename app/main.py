from fastapi import FastAPI
from assess_api import router as assess_router
from predict_api import router as predict_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount endpoints
app.include_router(predict_router, prefix="/predict", tags=["Prediction"])
app.include_router(assess_router, prefix="/assess", tags=["Assessment"])

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Nibras API. Use /predict or /assess."}

