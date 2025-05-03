from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from assess_api import router as assess_router
from predict_api import router as predict_router

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routers without prefixes
app.include_router(predict_router, tags=["predict"])
app.include_router(assess_router, tags=["assess"])
