import streamlit as st
import pandas as pd
import weaviate
from weaviate.auth import Auth
from datetime import datetime
import os
from groq import Groq
from google import genai
import json

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if 'collection_name' not in st.session_state:
    st.session_state.collection_name = None
if 'df' not in st.session_state:
    st.session_state.df = None

def generate_embeddings(text):
    try:
        result = genai_client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=text
        )
        return result.embeddings[0]
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

def create_weaviate_collection(collection_name):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY"))
    )
    
    schema = {
        "class": collection_name,
        "properties": [
            {"name": "text", "dataType": ["text"]},
            {"name": "unique_id", "dataType": ["text"]}
        ],
        "vectorizer": "none"
    }
    
    if not client.collections.exists(collection_name):
        client.collections.create(schema)
    
    return client

def process_csv(file, unique_id=None):
    try:
        df = pd.read_csv(file)
        st.session_state.df = df  
        
        csv_name = file.name.split('.')[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        collection_name = f"{csv_name}_{timestamp}"
        st.session_state.collection_name = collection_name  
        
        client = create_weaviate_collection(collection_name)
        collection = client.collections.get(collection_name)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        with collection.batch.dynamic() as batch:
            total_rows = len(df)
            for idx, row in df.iterrows():
                formatted_text = " | ".join([f"{str(value)} - {col}" for col, value in row.items()])
                embedding = generate_embeddings(formatted_text)
                
                if embedding:
                    properties = {
                        "text": formatted_text,
                        "unique_id": unique_id if unique_id else str(idx)
                    }
                    batch.add_object(properties=properties, vector=embedding)
                
                progress = (idx + 1) / total_rows
                progress_bar.progress(progress)
                status_text.text(f"Processing row {idx + 1}/{total_rows}")
        
        status_text.text("Processing complete!")
        return True
        
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return False
    finally:
        if 'client' in locals():
            client.close()

def handle_numbers(query):
    """Tool to handle numerical queries by generating and executing Python code"""
    try:
        df = st.session_state.df
        
        prompt = f"""
        Given the following query: "{query}"
        Generate only a Python code snippet using pandas to answer this query.
        The dataframe is called 'df'.
        Return only the code, nothing else.
        """
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a Python code generator. Generate only code, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        code = response.choices[0].message.content.strip()
        
        with st.expander("View Generated Code"):
            st.code(code, language="python")
        
        result = eval(code)
        
        if isinstance(result, (pd.DataFrame, pd.Series)):
            return result.to_string()
        return str(result)
        
    except Exception as e:
        return f"Error processing numerical query: {str(e)}"

def handle_text(query):
    """Tool to handle text queries by searching vector database for relevant information"""
    try:
        collection_name = st.session_state.collection_name
        
        query_embedding = genai_client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=query
        ).embeddings[0]
        
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY"))
        )
        
        collection = client.collections.get(collection_name)
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=3
        )
        
        if response.objects:
            contexts = [obj.properties["text"] for obj in response.objects]
            return "\n\n".join(contexts)
        return "No relevant information found."
        
    except Exception as e:
        return f"Error processing text query: {str(e)}"
    finally:
        if 'client' in locals():
            client.close()

def process_query_with_tools(query):
    """Process user query using LLM with tool calling capabilities"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "handle_numbers",
                "description": "For numerical queries that require calculations, aggregations, filtering by numbers, or statistical analysis on the dataset",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's numerical query to process"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "handle_text",
                "description": "For text-based queries that require searching for information, similar items, or contextual understanding of the dataset",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's text query to process"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a smart assistant for a data analysis application. Use the appropriate tool to answer user queries about their data."},
            {"role": "user", "content": query}
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0.1
    )

    message = response.choices[0].message
    
    if hasattr(message, 'tool_calls') and message.tool_calls:
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        st.info(f"Using {tool_name} to process your query...")
        
        if tool_name == "handle_numbers":
            result = handle_numbers(tool_args.get("query", query))
        elif tool_name == "handle_text":
            result = handle_text(tool_args.get("query", query))
        else:
            result = "Unknown tool called"
            
        final_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful data analysis assistant. Provide a clear, concise answer based on the tool's result."},
                {"role": "user", "content": query},
                {"role": "assistant", "content": "I'll help you with that question."},
                {"role": "function", "name": tool_name, "content": result},
            ],
            temperature=0.1
        )
        
        return {
            "tool_used": tool_name,
            "raw_result": result,
            "formatted_answer": final_response.choices[0].message.content
        }
    else:
        return {
            "tool_used": "none",
            "raw_result": "No specific tool was selected",
            "formatted_answer": message.content
        }

def main():
    st.title("CSV Query Agent")
    st.markdown("Upload a CSV file and ask questions about your data!")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    unique_id = st.text_input("Optional: Enter a unique ID for this dataset")
    
    if uploaded_file and st.button("Process CSV"):
        with st.spinner("Processing CSV file..."):
            if process_csv(uploaded_file, unique_id):
                st.success("CSV processed successfully!")
                st.session_state.file_processed = True
    
    if st.session_state.get('collection_name') and st.session_state.get('df') is not None:
        st.markdown("---")
        st.subheader("Ask Questions About Your Data")
        query = st.text_input("Enter your query")
        
        if query and st.button("Get Answer"):
            with st.spinner("Processing query..."):
                result = process_query_with_tools(query)
                
                st.markdown("### Answer:")
                st.write(result["formatted_answer"])
                
                with st.expander("View Technical Details"):
                    st.markdown(f"**Tool Used:** {result['tool_used']}")
                    st.markdown("**Raw Result:**")
                    st.code(result["raw_result"])

if __name__ == '__main__':
    main()