#!/usr/bin/env python3
"""
Test script for the RequirementEvaluatorAgent
"""

import asyncio
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.tools.evaluatior import RequirementEvaluatorAgent


async def test_evaluator():
    """Test the evaluator with the idea_sample.md content"""
    
    requirements = """Project Goal: Xây dựng web app quản lý đơn hàng giúp nhân viên kho xử lý ít nhất 1.000 đơn/ngày với thời gian phản hồi trung bình ≤ 150 ms.

Target Users: Nhân viên kho, quản lý kho (khoảng 20–30 người dùng nội bộ).

Core Features:
- Dashboard tổng quan đơn hàng (lọc theo trạng thái, ngày tạo)
- Chức năng "xác nhận đơn" & "đóng gói"
- Thông báo email/SMS khi đơn thay đổi trạng thái

Non-functional Constraints: Thời gian phản hồi API ≤ 150 ms, khả năng chịu tải 500 requests/giây, tuân thủ bảo mật OAuth2.

Success Metrics: Giảm thời gian xử lý đơn từ 5 phút xuống còn ≤ 2 phút, duy trì uptime ≥ 99,9% hàng tháng."""

    try:
        print("Initializing RequirementEvaluatorAgent...")
        evaluator = RequirementEvaluatorAgent()
        
        print("Evaluating requirements...")
        result = await evaluator.evaluate(requirements)
        
        print("\n=== EVALUATION RESULTS ===")
        print(f"Overall Score: {result.overall_score:.1f}/5")
        print("\nScores by Criterion:")
        for criterion, score in result.scores.items():
            print(f"- {criterion.replace('_', ' ').title()}: {score}/5")
        
        if result.questions:
            print("\nQuestions for Improvement:")
            for question in result.questions:
                print(f"- {question.question}")
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_evaluator())
