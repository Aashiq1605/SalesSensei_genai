from flask import Flask, render_template, jsonify, request
from flask_pymongo import PyMongo
import openai
import os
import pandas as pd
import pdfplumber
from pymongo import MongoClient
import gridfs
import uuid
##from Key import OPENAI_API_KEY
from graph_utils import generate_graph  # Importing the graph utility
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Retrieve API key
openai.api_key = os.getenv("OPENAI_API_KEY")


app = Flask(__name__)
app.config["MONGO_URI"] = os.getenv("MONGODB_URI")
app.config["UPLOAD_FOLDER"] = "./uploads"  # Folder to save uploaded files
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)  # Create folder if it doesn't exist
mongo = PyMongo(app)
fs = gridfs.GridFS(mongo.db)  # Initialize GridFS for file storage

# Store file data in memory for simplicity (can be extended to database storage)
file_data_store = {}

def extract_graph_details(question):
    """
    Query ChatGPT to extract details for graph generation from the user's question.
    """
    try:
        # Query OpenAI's API for extracting graph details
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"Analyze the following query and extract graph details, including graph type, x-axis, y-axis, and any additional information needed: {question}"}
            ],
            temperature=0.5,
            max_tokens=200,
        )
        # Parse the response from ChatGPT
        graph_details = response.choices[0].message.content.strip()
        print(f"Extracted Graph Details: {graph_details}")
        return eval(graph_details)  # Convert the string response to a Python dictionary
    except Exception as e:
        print(f"Error in extract_graph_details: {e}")
        return None


@app.route("/")
def home():
    try:
        # Retrieve chats from MongoDB
        chats = mongo.db.chats.find({})
        myChats = [chat for chat in chats]
        print("Chats retrieved successfully:", myChats)
        return render_template("new.html", myChats=myChats)
    except Exception as e:
        print(f"Error retrieving chats: {e}")
        return render_template("new.html", myChats=[])


@app.route("/api", methods=["POST"])
def qa():
    if request.method == "POST":
        try:
            print("Request received:", request.json)
            question = request.json.get("question")
            file_id = request.json.get("file_id")  # This may be None

            if not question:
                return jsonify({"answer": "No question provided"}), 400

            # Handle graph-related keywords in the question
            if "graph" in question.lower() or "chart" in question.lower():
                graph_details = extract_graph_details(question)
                if not graph_details:
                    return jsonify({"answer": "Could not understand the graph request. Please try again."}), 400

                graph_info = {
                    "graph_type": graph_details.get("graph_type", "bar"),
                    "x_label": graph_details.get("x_label", "X-axis"),
                    "y_label": graph_details.get("y_label", "Y-axis"),
                    "title": graph_details.get("title", "Generated Graph"),
                }

                if file_id and file_id in file_data_store:
                    graph_data = file_data_store[file_id]
                    graph_info["graph_data"] = graph_data
                    response = app.test_client().post("/generate_graph", json=graph_info)
                    return response
                else:
                    return jsonify({"answer": "No file data available for graph generation."}), 400

            # Retrieve file content if file_id is provided
            if file_id and file_id in file_data_store:
                file_data = file_data_store[file_id]
                file_type = file_data.get("type")

                if file_type == "csv":
                    # Use the CSV summary in the prompt
                    file_content = f"Here is the summary of the uploaded file '{file_id}':\n{file_data['ai_summary']}"
                elif file_type == "pdf":
                    # Use the PDF summary and suggestions in the prompt
                    file_content = (
                        f"Here is the summary of the uploaded PDF file '{file_id}':\n{file_data['ai_summary']}\n\n"
                        f"Suggestions for improvement:\n{file_data['improvement_suggestions']}"
                    )
                else:
                    file_content = f"The uploaded file '{file_id}' type is unsupported for detailed analysis."

                # Combine file content with the user's question
                full_prompt = f"{file_content}\n\nUser's question: {question}"
            else:
                # No file context available, use the question directly
                full_prompt = question

            # Query OpenAI with the constructed prompt
            print("Querying ChatGPT with prompt:", full_prompt)
            response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        """🚀 SalesSensei: Your AI-Powered Sales Coach & Deal Closer
                    You are SalesSensei, an AI-driven sales strategist built for AnyMind Group, designed to boost win rates in negotiations, optimize client proposals, and improve overall sales performance. Your expertise is rooted in best-in-class business development (BD) practices, helping sales professionals prepare effectively for client meetings, pitch presentations, and strategic deal-making.

                    🔹 What SalesSensei Can Do (Key Capabilities)
                    📊 Proposal & Document Review:

                    Analyze sales decks, pitch presentations, and BD documents to highlight unclear points, client concerns, and areas needing improvement.
                    Structure feedback based on winning sales narratives, logical flow, and high-impact messaging.
                    User-Specified Review Requests: Users can request feedback on specific aspects like clarity, objection handling, or proposal structuring.
                    🤝 Client Meeting & Negotiation Prep:

                    Guide users on key talking points, strategic positioning, and rebuttal techniques based on the provided context.
                    Help structure a persuasive sales story that eliminates client doubts and increases conversion likelihood.
                    Provide a high-quality hypothesis for each meeting to improve sales efficiency and shorten prep time.
                    📈 Data-Driven Sales Insights & Visualization:

                    Convert sales data into meaningful graphs and trends for better decision-making.
                    Identify patterns in deal performance and highlight key takeaways.
                    🔹 SalesSensei’s Core Approach
                    ✅ Efficiency: Helps sales professionals prepare faster & smarter for meetings.
                    ✅ Objectivity: Provides structured, data-backed insights to refine proposals.
                    ✅ Scalability: Empowers junior reps by making elite sales knowledge accessible.

                    🔹 SalesSensei’s Response Formatting
                    Use double line breaks between key points for readability.
                    If a user uploads a file, acknowledge receipt and ask for clarification on the specific review focus.
                    If reviewing a proposal or sales deck, provide structured feedback covering:
                    1️⃣ Clarity & Conciseness: Are key messages easy to understand?
                    2️⃣ Client Concerns: What potential objections might arise?
                    3️⃣ Impact & Persuasiveness: How compelling is the proposal?
                    4️⃣ Visual & Structural Issues: Are slides or documents logically structured?
                    🔹 SalesSensei’s Behavior & Personality
                    Professional & Tactical: Like a top-tier BD mentor, responses should be sharp, direct, and highly actionable.
                    Engaging & Motivating: Encourages users to refine their approach with constructive, insightful feedback.
                    Adaptive & Context-Aware: Adjusts responses based on user input, ensuring tailored advice.
                    🔹 Special Instructions
                    1️⃣ If a user greets SalesSensei, respond with:

                    A short, engaging intro with business-related emojis.
                    A four-line summary of how SalesSensei helps close deals & improve BD performance.
                    2️⃣ If a file is uploaded, respond with:

                    Acknowledgment ("Received your document! How would you like me to help?")
                    Follow-up question ("Do you want an overall review, or feedback on specific aspects like clarity, objections, or impact?")
                    3️⃣ If a data visualization request is made, respond by:

                    Confirming the graph type & attributes before generating it.
                    Providing a concise summary of the insights from the graph.
                    Example Responses
                    📌 User Uploads a Proposal Deck
                    👉 “I’ve analyzed your proposal. Here’s structured feedback:
                    1️⃣ Clarity: The core message is strong, but the value proposition on Slide 3 needs to be sharper.
                    2️⃣ Objections: Clients may question pricing flexibility—consider adding an ROI comparison.
                    3️⃣ Persuasiveness: Slide 7 is impactful, but a real-world success case would strengthen it.”_

                    📌 User Asks for Meeting Prep Advice
                    👉 "For your upcoming client meeting, focus on these points:
                    🔹 Key Client Pain Points: Emphasize cost savings & operational efficiency.
                    🔹 Anticipated Objections: Expect concerns about implementation time—have a fast-track plan ready.
                    🔹 High-Impact Strategy: Use social proof & competitor comparisons to reinforce credibility.”_

                    📌 User Asks for a Sales Data Graph
                    👉 “Here’s your requested revenue vs. conversion rate chart. Key takeaway: Revenue spiked 20% after increasing personalized follow-ups—worth replicating next quarter. 📊”
                        If a data given analyse each column and rows
                    🔹 AnyMind Group Core Values Alignment
                    SalesSensei aligns with AnyMind Group’s principles:
                    🏆 Be Bold: Encourages sales reps to confidently refine their pitch & close deals.
                    🤝 Achieve Together: Empowers BD teams by making elite sales expertise accessible.
                    📈 Stay Updated: Delivers insights based on best BD practices & trends.
                    ⚡ Move Faster: Helps users prepare high-quality pitches in minimal time.
                    💡 Be Open: Allows customized requests to adapt responses to user needs.

                    📌 Why This Prompt Is Effective
                    ✅ Highly Structured & Goal-Oriented: SalesSensei doesn’t just answer—it guides users through the sales process.
                    ✅ File Analysis Acknowledgment: Ensures uploaded documents are processed with context-driven feedback.
                    ✅ Dynamic Meeting Preparation: Tailors responses based on user-provided details & sales scenarios.
                    ✅ Business-Ready Language: Delivers professional, precise, and impactful insights.

                    🚀 Final Outcome
                    This version of SalesSensei will function like a top-tier BD mentor, providing structured guidance to optimize deal-making, refine proposals, and enhance sales efficiency—all while ensuring AI-generated insights align with real-world BD expertise.
                    
                    Add emojis in each chat,
                    If I upload a file I want you to analyse and we will chat based on that file data.
                    """
                    ),
                },
                {"role": "user", "content": full_prompt},  # Ensure this is a string
            ],
                temperature=0.7,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            # Extract OpenAI response
            answer = response.choices[0].message.content.strip()
            print(f"ChatGPT Response: {answer}")


            # Save the question and answer to the database
            mongo.db.chats.insert_one({"question": question, "answer": answer})
            return jsonify({"question": question, "answer": answer})

        except Exception as e:
            print(f"Error processing question: {e}")
            return jsonify({"answer": "An error occurred while processing your request. Please try again later."}), 500



@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file:
        try:
            # Use filename as a simple file ID (update for production if needed)
            file_id = str(uuid.uuid4()) 
            file_data_store[file_id] = {}  # Create a data store entry for the file
            print(f"File received with ID: {file_id}")

            # Process CSV files
            if file.filename.endswith(".csv"):
                file.seek(0)  # Reset file pointer
                try:
                    # Read CSV into a pandas DataFrame
                    data = pd.read_csv(file)

                    # Check if the data is valid
                    if data.empty or data.columns.empty:
                        return jsonify({"error": "The uploaded CSV file is empty or invalid."}), 400

                    # Log columns and preview data for debugging
                    print(f"CSV Columns: {list(data.columns)}")
                    print(data.head())  # ✅ Print the first few rows

                    # Convert the first 30 rows to plain text for summarization
                    summary_input = data.head(30).to_string(index=False)

                    # Query OpenAI for summarization and insights
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are a data analyst."},
                                {"role": "user", "content": f"Here is a sample of the uploaded data:\n\n{summary_input}\n\nPlease summarize and analyze this data. Highlight trends, insights, and any anomalies."}
                            ],
                            temperature=0.7,
                            max_tokens=300,
                        )
                        ai_summary = response.choices[0].message.content.strip()
                    except Exception as e:
                        print(f"Error querying OpenAI for CSV analysis: {e}")
                        ai_summary = "Unable to analyze the data using OpenAI."

                    # ✅ Store both column names AND actual data
                    file_data_store[file_id] = {
                        "type": "csv",
                        "name":file.filename,
                        "columns": list(data.columns),
                        "data": data.to_dict(orient="list"),  # ✅ Store actual data
                        "ai_summary": ai_summary
                    }

                    # ✅ Debugging: Print stored data (including actual values)
                    print(f"Stored File Data for {file_id}: {file_data_store[file_id]}")

                    return jsonify({
                        "message": "File uploaded successfully",
                        "file_id": file_id,
                        "file_type": "CSV",
                        "file_name": file.filename,
                        "columns": list(data.columns),
                        "ai_summary": ai_summary
                    }), 200

                except Exception as e:
                    print(f"Error reading CSV file: {e}")
                    return jsonify({"error": "Failed to read the CSV file. Ensure it is correctly formatted."}), 400

            
            # ✅ Handle PDF Uploads
            elif file.filename.endswith(".pdf"):
                file.seek(0)  # Reset file pointer
                try:
                    with pdfplumber.open(file) as pdf:
                        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                    
                    if not text.strip():
                        print("Error: PDF file contains no readable text.")
                        return jsonify({"error": "The uploaded PDF file is empty or contains no readable text."}), 400

                    print(f"✅ Extracted PDF Text Length: {len(text)} characters")

                    # ✅ Truncate text if it exceeds OpenAI's token limits
                    max_chars = 2000
                    truncated_text = text[:max_chars]

                    # ✅ Query OpenAI for summarization
                    try:
                        summary_response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are a document summarizer."},
                                {"role": "user", "content": f"Summarize the following document:\n\n{truncated_text}"}
                            ],
                            temperature=0.7,
                            max_tokens=300,
                        )
                        ai_summary = summary_response.choices[0].message.content.strip()

                        # ✅ Generate improvement suggestions
                        improvement_response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are a business consultant specializing in improving proposals."},
                                {"role": "user", "content": f"Based on this summary:\n\n{ai_summary}\n\nSuggest improvements to make this proposal more effective and persuasive."}
                            ],
                            temperature=0.7,
                            max_tokens=300,
                        )
                        improvement_suggestions = improvement_response.choices[0].message.content.strip()

                    except Exception as e:
                        print(f"Error querying OpenAI for PDF analysis: {e}")
                        ai_summary = "Unable to generate a summary using OpenAI."
                        improvement_suggestions = "Unable to generate improvement suggestions using OpenAI."

                    # ✅ Store PDF data
                    file_data_store[file_id]["type"] = "pdf"
                    file_data_store[file_id]["name"] = file.filename
                    file_data_store[file_id]["text"] = text
                    file_data_store[file_id]["ai_summary"] = ai_summary
                    file_data_store[file_id]["improvement_suggestions"] = improvement_suggestions
                    print(f"✅ Stored PDF File Data for {file_id}")
                    print(f"filefile name is {file.filename}")
                    return jsonify({
                        "message": "File uploaded successfully",
                        "file_id": file_id,
                        "file_type": "PDF",
                        "name":file.filename,
                        "ai_summary": ai_summary,
                        "improvement_suggestions": improvement_suggestions
                    }), 200
                        
                except Exception as e:
                    print(f"Error reading PDF file: {e}")
                    return jsonify({"error": "Failed to read the PDF file. Ensure it contains readable text."}), 400

             
            else:
                return jsonify({"error": "Unsupported file type. Only CSV and PDF are allowed."}), 400

        except Exception as e:
            print(f"Error processing file: {e}")
            return jsonify({"error": "An error occurred while processing the file."}), 500


@app.route("/generate_graph", methods=["POST"])
def generate_graph_endpoint():
    try:
        # Parse the incoming request data
        data = request.json
        print(f"Graph generation payload: {data}")

        # Extract required fields
        graph_type = data.get("graph_type", "bar")
        file_id = data.get("file_id")  # Retrieve the file_id from the payload
        x_label = data.get("x_label", "X-axis")
        y_label = data.get("y_label", "Y-axis")
        title = data.get("title", "Generated Graph")

        # Validate file_id
        if not file_id or file_id not in file_data_store:
            print(f"Error: File ID '{file_id}' is invalid or data not found.")
            return jsonify({"error": "Invalid file ID or file data not found."}), 400

        # Retrieve data from file_data_store
        graph_data = file_data_store[file_id]
        print(f"Retrieved graph data: {graph_data}")

        # ✅ Ensure 'data' key exists
        if "data" not in graph_data:
            print(f"Error: No 'data' key found in graph_data for file ID {file_id}")
            return jsonify({"error": "No valid data found in the uploaded file."}), 400

        data_dict = graph_data["data"]  # Extract the actual data dictionary

        # ✅ Ensure both X and Y labels exist in the dataset
        if x_label not in data_dict or y_label not in data_dict:
            print(f"Error: Attributes '{x_label}' or '{y_label}' not found in graph data.")
            return jsonify({"error": f"Attributes '{x_label}' or '{y_label}' not found in graph data."}), 400

        # ✅ Log column data for debugging
        print(f"Data for '{x_label}': {data_dict[x_label]}")
        print(f"Data for '{y_label}': {data_dict[y_label]}")

        # Generate the graph
        graph_base64 = generate_graph(graph_type, data_dict, title, x_label, y_label)

        # Return the generated graph as a Base64 string
        return jsonify({"graph": f"data:image/png;base64,{graph_base64}"})

    except Exception as e:
        # Catch and log unexpected errors
        print(f"Error in /generate_graph: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)