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

# Include routers
app.include_router(predict_router, prefix="/predict", tags=["predict"])
app.include_router(assess_router, prefix="/assess", tags=["assess"])

@app.get("/")
def root():
    return {"message": "Nibras API is running ✅"}
