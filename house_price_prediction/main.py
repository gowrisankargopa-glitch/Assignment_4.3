"""
FastAPI web application for housing price prediction
with SQLite database integration and dependency injection
"""

from fastapi import FastAPI, Depends, HTTPException, Path, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime
from typing import Optional, List

from house_price_prediction.database import init_db, get_db, InferenceRecord
from house_price_prediction.schemas import HousingInput, PredictionResponse, AllModelsResponse, InferenceHistoryResponse
from house_price_prediction.inference import HousingInference

# Initialize FastAPI app
app = FastAPI(
    title="Housing Price Prediction API",
    description="ML API for predicting house prices using Linear Regression and SVM models",
    version="1.0.0"
)

# Global inference engine
inference_engine = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize database and load models on application startup
    """
    global inference_engine
    
    print("Starting up Housing Price Prediction API...")
    
    # Initialize database
    init_db()
    
    # Load inference models
    try:
        inference_engine = HousingInference()
        print("✓ Inference models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading inference models: {e}")
        raise
    
    print("✓ Application startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on application shutdown
    """
    print("Shutting down Housing Price Prediction API...")


@app.get("/", tags=["Info"])
async def root():
    """
    Welcome endpoint with API documentation
    """
    return {
        "message": "Welcome to Housing Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "prediction": "/predict/{model_name}",
            "history": "/inferences/{model_name}",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "inference_engine_loaded": inference_engine is not None
    }


@app.post(
    "/predict/{model_name}",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Make a housing price prediction"
)
async def predict(
    model_name: str = Path(
        ...,
        description="Model to use: 'linear' or 'svm'",
        regex="^(linear|svm)$"
    ),
    input_data: HousingInput = None,
    db: Session = Depends(get_db)
):
    """
    Make a housing price prediction and store the record in the database
    
    Args:
        model_name: The model to use for prediction ('linear' or 'svm')
        input_data: Housing features as input
        db: Database session (injected via dependency)
    
    Returns:
        Prediction record with ID, input features, prediction value, and timestamp
    
    Raises:
        HTTPException: If model is invalid or prediction fails
    """
    if not inference_engine:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")
    
    if model_name not in ["linear", "svm"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model_name}. Use 'linear' or 'svm'"
        )
    
    try:
        # Make prediction
        prediction_value = inference_engine.predict(
            input_data.model_dump(),
            model_name=model_name
        )
        
        # Create database record
        record = InferenceRecord(
            longitude=input_data.longitude,
            latitude=input_data.latitude,
            housing_median_age=input_data.housing_median_age,
            total_rooms=input_data.total_rooms,
            total_bedrooms=input_data.total_bedrooms,
            population=input_data.population,
            households=input_data.households,
            median_income=input_data.median_income,
            ocean_proximity=input_data.ocean_proximity,
            model_name=model_name,
            prediction=prediction_value,
            timestamp=datetime.utcnow()
        )
        
        # Save to database
        db.add(record)
        db.commit()
        db.refresh(record)
        
        return record
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get(
    "/inferences/{model_name}",
    response_model=InferenceHistoryResponse,
    tags=["History"],
    summary="Get inference history for a model"
)
async def get_inference_history(
    model_name: str = Path(
        ...,
        description="Model name to filter records: 'linear' or 'svm'"
    ),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    offset: Optional[int] = Query(0, ge=0, description="Number of records to skip"),
    db: Session = Depends(get_db)
):
    """
    Get all previous inference records for a specific model
    
    Args:
        model_name: Filter by model name ('linear' or 'svm')
        limit: Maximum number of records to return (default: 100)
        offset: Number of records to skip for pagination (default: 0)
        db: Database session (injected via dependency)
    
    Returns:
        Historical inference records with total count
    
    Raises:
        HTTPException: If model is invalid or database query fails
    """
    if model_name not in ["linear", "svm"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model_name}. Use 'linear' or 'svm'"
        )
    
    try:
        # Query total count
        total_count = db.query(InferenceRecord).filter(
            InferenceRecord.model_name == model_name
        ).count()
        
        # Query paginated records
        records = db.query(InferenceRecord).filter(
            InferenceRecord.model_name == model_name
        ).order_by(
            desc(InferenceRecord.timestamp)
        ).offset(offset).limit(limit).all()
        
        return InferenceHistoryResponse(
            total_records=total_count,
            model_name=model_name,
            records=records
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")


@app.post(
    "/predict-all",
    response_model=dict,
    tags=["Prediction"],
    summary="Make predictions with multiple models"
)
async def predict_all(
    input_data: HousingInput,
    db: Session = Depends(get_db)
):
    """
    Make predictions using all available models and store records
    
    Args:
        input_data: Housing features as input
        db: Database session (injected via dependency)
    
    Returns:
        Dictionary with predictions from all models
    """
    if not inference_engine:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")
    
    try:
        # Get predictions from all models
        predictions = inference_engine.predict(
            input_data.model_dump(),
            model_name="all"
        )
        
        # Store each prediction in database
        for model_name, prediction_value in predictions.items():
            record = InferenceRecord(
                longitude=input_data.longitude,
                latitude=input_data.latitude,
                housing_median_age=input_data.housing_median_age,
                total_rooms=input_data.total_rooms,
                total_bedrooms=input_data.total_bedrooms,
                population=input_data.population,
                households=input_data.households,
                median_income=input_data.median_income,
                ocean_proximity=input_data.ocean_proximity,
                model_name=model_name.split("_")[0],  # Extract 'linear' or 'svm'
                prediction=prediction_value,
                timestamp=datetime.utcnow()
            )
            db.add(record)
        
        db.commit()
        
        return {
            "input_data": input_data,
            "predictions": predictions,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
