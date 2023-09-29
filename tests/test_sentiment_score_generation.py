import unittest
import asyncio
from sentiment_score_generation import (
    generate_all_prompts,
    generate_llm_sentiment_score_prompt,
    combine_populated_prompts_with_source_text,
    validate_llm_generated_sentiment_response,
    run_llm_in_process,
    parallel_attempt,
    combine_llm_generated_sentiment_responses,
    analyze_focus_area_sentiments
)

class TestSentimentScoreGeneration(unittest.TestCase):

    def setUp(self):
        self.focus_key = "financial_investor_focus"
        self.scoring_scale_explanation = "Test explanation"
        self.source_text_positive = "The company has shown impressive growth this year."
        self.source_text_negative = "The company's performance has been disappointing."
        self.model_name = "Test model"

    def test_generate_all_prompts(self):
        populated_prompts = generate_all_prompts(self.focus_key, self.scoring_scale_explanation)
        self.assertIsInstance(populated_prompts, dict)
        self.assertTrue('Optimistic' in populated_prompts)

    def test_generate_llm_sentiment_score_prompt(self):
        prompt = generate_llm_sentiment_score_prompt('Optimistic', 'Positive outlook', 'Investors', self.scoring_scale_explanation)
        self.assertIsInstance(prompt, str)

    def test_combine_populated_prompts_with_source_text(self):
        combined = combine_populated_prompts_with_source_text('Test prompt', 'Test source')
        self.assertIsInstance(combined, str)

    def test_validate_llm_generated_sentiment_response(self):
        sentiment_score, justification = validate_llm_generated_sentiment_response('5 | Test justification', -10, 10)
        self.assertIsInstance(sentiment_score, float)
        self.assertIsInstance(justification, str)

    def test_run_llm_in_process(self):
        result = run_llm_in_process('Test prompt', 'Test model')
        self.assertIsInstance(result, dict)

    def test_parallel_attempt(self):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(parallel_attempt('Test prompt', 'Test model'))
        self.assertIsInstance(result, str)

    def test_combine_llm_generated_sentiment_responses(self):
        outputs = ['5 | Justification', '6 | Justification']
        mean_score, ci, iqr, iqr_pct, justifications = combine_llm_generated_sentiment_responses(outputs, -10, 10)
        self.assertIsInstance(mean_score, float)
        self.assertIsInstance(ci, list)
        self.assertIsInstance(iqr, list)
        self.assertIsInstance(iqr_pct, float)
        self.assertIsInstance(justifications, list)

    def test_analyze_focus_area_sentiments(self):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(analyze_focus_area_sentiments(self.focus_key, self.scoring_scale_explanation, self.source_text, self.model_name))
        self.assertIsInstance(result, dict)

    def test_positive_sentiment(self):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(analyze_focus_area_sentiments(self.focus_key, self.scoring_scale_explanation, self.source_text_positive, self.model_name))
        self.assertIsInstance(result, dict)
        self.assertGreater(result['individual_sentiment_report_dict']['Optimistic']['sentiment_scores_dict']['mean_sentiment_score'], 50)

    def test_negative_sentiment(self):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(analyze_focus_area_sentiments(self.focus_key, self.scoring_scale_explanation, self.source_text_negative, self.model_name))
        self.assertIsInstance(result, dict)
        self.assertLess(result['individual_sentiment_report_dict']['Optimistic']['sentiment_scores_dict']['mean_sentiment_score'], -50)
        
if __name__ == '__main__':
    unittest.main()
