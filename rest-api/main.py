from fastapi import FastAPI, UploadFile,Form
from fastapi.middleware.cors import CORSMiddleware

app_ganteng = FastAPI()

app_ganteng.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app_ganteng.post("/recommend")
async def recommend_songs(form: Form):
    pass

