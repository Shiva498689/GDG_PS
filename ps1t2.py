#please use ps2venv for running this file as the virtual env is named incorrectly by mistake
import streamlit as st
import warnings
import logging
import yfinance as yf
import json
from groq import Groq
from ddgs import DDGS

# Try and Catch Blocks are used to prevent the fall of code in between without any error mmessage

st.set_page_config(page_title="Stock Market and Investment Analyst Tool", layout="wide")


warnings.simplefilter("ignore") # ignore warnings
logging.getLogger("jupyter_client").setLevel(logging.ERROR)

st.title("Investment and Stock Market Analyst ")


groq_api_key = "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" # please do not use this as i will be deleting it just after the submission


def stock_price(ticker): # function to fetch the stock data using yfinance
 
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if data.empty: 
            return json.dumps({"error": f"No data for {ticker}"})
        
        price = data['Close'].iloc[-1]
        return json.dumps({"ticker": ticker, "price": round(price, 2), "currency": "USD"})
    except Exception as e:
        return json.dumps({"error": str(e)})

def trending_news(query):# function to capture trending news and sentiment for the company 
   
    try:
        results = DDGS().text(query, max_results=5)
        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})

def Display_stock_line_graph(ticker):
    
    try:
        period="1y"
        data = yf.Ticker(ticker).history(period="1y")
        if data.empty: 
            return json.dumps({"error": f"No data for {ticker}"})
        
        st.subheader(f"{period} Trend: {ticker.upper()}")
        st.line_chart(data['Close'])
        
        return json.dumps({
            "status": "success", 
            "message": f"Chart for {ticker} plotted successfully. Tell the user you have displayed it."
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

# function map
available_functions = {
    "stock_price": stock_price,
    "trending_news": trending_news,
    "Display_stock_line_graph": Display_stock_line_graph
}

# tools for making it agentic
tools = [
    {
        "type": "function",
        "function": {
            "name":  "stock_price",
            "description": "Get current stock price.",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trending_news",
            "description": "Search for latest financial and stock market news.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Display_stock_line_graph",
            "description": "Plot a line chart  of stock history.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "ticker": {"type": "string", "description": "Stock symbol (e.g. AAPL)"}, 
                }, 
                "required": ["ticker"]
            },
        },
    }
]

# defining prompt for Groq

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system", 
            "content": (
                "You are a financial analyst helper. "
                "You have access to tools: 'stock_price', 'trending_news', and 'Display_stock_line_graph'. "
                "IMPORTANT: When you need to use a tool, use the standard tool calling mechanism. "
                "DO NOT output raw XML or text like <function=...>. "
                "If you plot a chart, simply tell the user 'I have displayed the chart' after calling the tool."
            )
        }
    ]


for message in st.session_state.messages:
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
    else:
        role = message.role
        content = message.content

    if role == "system" or role == "tool":
        continue
    
    if content:
        with st.chat_message(role):
            st.write(content) # display the chat history like other chatbots 


if user_input := st.chat_input("Ask about stock market and stock of either national or international companies ."):
    

    
    client = Groq(api_key=groq_api_key)

    #append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
        
        # '' I am doing the api call 2 times such that 1st time it fetches the user query and use the tools as per 
        # itself than it use the tools to get the results and than go to the final answer after analysing them all''

   
    with st.chat_message("assistant"):
        try:
            
            system_prompt = st.session_state.messages[0]
            
            recent_history = st.session_state.messages[1:][-10:] 
            messages_for_api = [system_prompt] + recent_history

     
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",# using ,model with 70B params
                messages=messages_for_api, # Uses sliced history
                tools=tools,
                tool_choice="auto"
            )
            
            response_msg = response.choices[0].message 
            tool_calls = response_msg.tool_calls


            if tool_calls:
               
                st.session_state.messages.append(response_msg)
                
                with st.status("Thinking...", expanded=True) as status:
                    for tool_call in tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)
                        
                        status.write(f"Reasoning: `{func_name}`")
                        
                        if func_name in available_functions:
                            func_result = available_functions[func_name](**func_args)
                            
                            # append result to history
                            st.session_state.messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": func_name,
                                "content": func_result,
                            })
             
                recent_history = st.session_state.messages[1:][-10:] # fetching recent 10 messages so that model not goes outside the context window of Groq
                final_messages_for_api = [system_prompt] + recent_history
                
               
                final_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=final_messages_for_api
                )
                
                final_ans = final_response.choices[0].message.content
                st.write(final_ans)
                st.session_state.messages.append({"role": "assistant", "content": final_ans})

       
            else:
                st.write(response_msg.content)
                st.session_state.messages.append(response_msg)

        except Exception as e:
            st.error(f"Error: {e}")
