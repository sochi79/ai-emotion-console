#!/usr/bin/env python3
"""
AI Emotion - Console Application
Run on Railway: python app.py
"""

import sys
import os

def main():
    print("🤖 AI EMOTION - CONSOLE APPLICATION")
    print("="*40)
    print("1. Collect Training Data")
    print("2. Train Model")
    print("3. Run Emotion Detection")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        from data_collector import DataCollector
        collector = DataCollector()
        collector.run_collection_session()
        
    elif choice == "2":
        from model_trainer import ModelTrainer
        trainer = ModelTrainer()
        trainer.train_model()
        
    elif choice == "3":
        print("🚧 Emotion Detection - Coming Soon!")
        # Здесь будет код для распознавания эмоций
        
    elif choice == "4":
        print("👋 Thank you for using AI Emotion!")
        sys.exit(0)
        
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()