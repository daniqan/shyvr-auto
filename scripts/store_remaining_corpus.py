#!/usr/bin/env python3
"""
Store remaining corpus data that failed due to numeric overflow
"""

import asyncio
import pickle
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline.multi_granularity_collector import MultiGranularityCollector
import structlog

logger = structlog.get_logger(__name__)

async def store_remaining_data():
    """Load checkpoint and store remaining data"""
    
    checkpoint_file = Path("checkpoint_corpus_collection.pkl")
    
    if not checkpoint_file.exists():
        print("No checkpoint file found. The collection may have completed.")
        return
    
    print("Loading checkpoint data...")
    with open(checkpoint_file, 'rb') as f:
        corpus_data = pickle.load(f)
    
    print(f"Found {len(corpus_data)} datasets in checkpoint")
    
    # Filter for PEPE data only (since others were stored successfully)
    pepe_data = {k: v for k, v in corpus_data.items() if k.startswith('PEPE')}
    
    if not pepe_data:
        print("No PEPE data found in checkpoint")
        return
    
    print(f"Found {len(pepe_data)} PEPE datasets to store:")
    for key in pepe_data:
        print(f"  - {key}: {len(pepe_data[key])} records")
    
    # Initialize collector
    collector = MultiGranularityCollector()
    
    try:
        print("\nStoring PEPE data to database...")
        await collector.store_corpus_to_database(pepe_data)
        print("✅ Successfully stored PEPE data!")
        
        # Clean up checkpoint
        checkpoint_file.unlink()
        print("Checkpoint file removed")
        
    except Exception as e:
        print(f"❌ Error storing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(store_remaining_data())