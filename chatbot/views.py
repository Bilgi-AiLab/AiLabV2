import os
from openai import OpenAI
from django.http import JsonResponse
import json

def chatbot(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "")

            if not user_message:
                return JsonResponse({"error": "Message is required"}, status=400)

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key="api-key",
            )

            completion = client.chat.completions.create(

            model="deepseek/deepseek-r1-distill-llama-70b:free",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant in a web application. This web application consists of applying textual data analysis such as sentiment analysis, summarization, topic modeling and document similarity. There is also a documentation section which covers everything. The user will provide a series of questions or small talk with you and your task is to provide detailed and accurate responses based on the context provided."
                },
                {
                    "role": "user",
                    "content": f"{user_message}"
                }
            ]
            )
            bot_reply = completion.choices[0].message.content
            return JsonResponse({"response": bot_reply})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)
