from google import genai

client = genai.Client(api_key="AIzaSyD5RCulju2gabG2iB_QjlD3TzDdDlm6uMQ")

response = client.models.generate_content(
    model="gemini-2.0-flash",   # 🔥 use latest model
    contents="ping"
)

print(response.text)