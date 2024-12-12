from fastapi import FastAPI, UploadFile,Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ml_models.models import load_model, load_encoder, predict
from typing import Annotated, List
import logging
import os
import numpy as np
import traceback

app_ganteng = FastAPI()

app_ganteng.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RecommendationLogger")

# Request payload schema
class RecommendationRequest(BaseModel):
    song_id_sequence: List[str]
    genre_id_sequence: List[List[str]]

genre_encoder = load_encoder(os.path.join("exported_models", "genre_encoder.pkl"))
song_encoder = load_encoder(os.path.join("exported_models", "song_encoder.pkl"))

model = load_model(os.path.join("exported_models", "gru4rec_model.keras"))

@app_ganteng.post("/recommend")
async def recommend_songs(request: RecommendationRequest):
    # Extract payload
    song_id_sequence = request.song_id_sequence
    genre_id_sequence = request.genre_id_sequence
    
    # Encode the song ID sequence
    try:
        encoded_song_id_sequence = song_encoder.transform(song_id_sequence)
    except Exception as e:
        logger.error(f"Failed to encode song ID sequence: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=400,
            content={
                "status": 400,
                "message": f"Failed to encode song ID sequence: {str(e)}",
            },
        )
    

    encoded_genre_sequence = []
    try:
        for genre in genre_id_sequence:
            # print(genre)
            encoded_genre_sequence.append(genre_encoder.transform(genre))
    except Exception as e:
        logger.error(f"Failed to convert inputs to numpy arrays: {traceback.format_exc()}")
        return JSONResponse(status_code=400, content={
                "status": 400,
                "message":f"Failed to encode genre sequence: {str(e)}",
            }
        )
    
    # Predict
    try:
        predicted_sequence = predict(model, encoded_song_id_sequence, encoded_genre_sequence, 10)
        
        # inverse transform the sequence
        predicted_sequence = song_encoder.inverse_transform(predicted_sequence)
        print(predicted_sequence)
    except Exception as e:
        logger.error(f"Prediction Failed: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={
                "status": 500,
                "message":f"Prediction failed: {str(e)}",
            }
        )
    
    # Return results
    return JSONResponse(status_code=200, content={
        "status": 200, 
        "message": "Recommendation given",
        "predicted_sequence": predicted_sequence.tolist()})
    
    

