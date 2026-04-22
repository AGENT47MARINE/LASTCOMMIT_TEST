import json
import time
import logging
from graph import app

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_runner")

def run_test_suite(file_path):
    with open(file_path, "r") as f:
        tests = json.load(f)
    
    logger.info(f"--- Starting Test Suite: {len(tests)} cases found ---")
    
    passed = 0
    for i, test in enumerate(tests):
        query = test["query"]
        category = test["category"]
        expected = test["expected"]
        
        logger.info(f"Test {i+1}/{len(tests)} | Category: {category}")
        logger.info(f"Query: {query}")
        
        start_time = time.time()
        initial_state = {
            "input": query,
            "intent": None,
            "result": None,
            "confidence": 0.0,
            "error": None,
            "steps": [],
            "retries": 0
        }
        
        try:
            final_state = app.invoke(initial_state)
            
            # Extract result string (simplified logic from api.py)
            res_dict = final_state.get("result", {})
            worker_res = res_dict.get("result", {}) if isinstance(res_dict, dict) else {}
            
            actual = None
            search_keys = ["answer", "solution", "summary", "analysis", "entities", "anomalies", "question"]
            for key in search_keys:
                if key in worker_res: actual = worker_res[key]; break
                if key in res_dict: actual = res_dict[key]; break
            
            if not actual: actual = str(res_dict)
            
            duration = time.time() - start_time
            logger.info(f"Actual: '{actual}' | Expected: '{expected}' | Time: {duration:.2f}s")
            
            # Simple substring or exact match check (evaluator uses cosine, we use logic)
            if str(expected).lower() in str(actual).lower():
                logger.info("RESULT: PASSED")
                passed += 1
            else:
                logger.warning("RESULT: FAILED")
                
        except Exception as e:
            logger.error(f"Error during test: {str(e)}")
        
        logger.info("-" * 40)

    logger.info(f"--- Test Suite Complete: {passed}/{len(tests)} Passed ---")

if __name__ == "__main__":
    run_test_suite("test_queries.json")
