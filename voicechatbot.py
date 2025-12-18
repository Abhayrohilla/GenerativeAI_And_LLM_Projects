import time
import re
from gtts import gTTS
import pygame
from io import BytesIO
import speech_recognition as sr


from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain



# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------


LISTENING_TIMEOUT = 10  # seconds
PHRASE_TIME_LIMIT = 8   # seconds


# Initialize pygame for audio playback
pygame.mixer.init()
pygame.mixer.music.set_volume(1.0)



# --------------------------------------------------
# SYSTEM PROMPT FOR PRACHI
# --------------------------------------------------


SYSTEM_PROMPT = """आप Prachi हैं - Kovon की friendly calling assistant।


RULES:
1. ONLY Hindi में बात करें (common English words OK - job, company, salary)
2. बहुत short और clear sentences बोलें (max 2-3 sentences at a time)
3. Professional पर friendly tone
4. हर response के END में status code:
   [702] = conversation जारी रखें
   [701] = call end करें


CONVERSATION STEPS:
1. Greeting + Kovon introduction
2. "क्या आपको overseas job में interest है?"
3. अगर हाँ → Name पूछें
4. Age पूछें
5. Education पूछें
6. Experience पूछें
7. "धन्यवाद, team contact करेगी" → [701]


IMPORTANT:
- एक बार में एक ही question पूछें
- User का जवाब सुनें फिर अगला question
- Short responses (10-15 words max)
- हमेशा [702] या [701] लगाएं


EXAMPLES:
"नमस्ते! मैं Kovon से Prachi हूँ। [702]"
"क्या आपको overseas jobs में interest है? [702]"
"बढ़िया! आपका नाम क्या है? [702]"
"धन्यवाद! हमारी team contact करेगी। [701]"
"""



# --------------------------------------------------
# TEXT-TO-SPEECH (GOOGLE TTS)
# --------------------------------------------------


def speak(text):
    """
    Convert text to speech using Google TTS
    Better Hindi pronunciation than pyttsx3
    """
    if not text or text.strip() == "":
        return
   
    clean_text = text.strip()
    print(f"\n🤖 Prachi: {clean_text}")
   
    try:
        # Generate speech
        tts = gTTS(text=clean_text, lang='hi', slow=False)
       
        # Save to BytesIO (in-memory, no file I/O)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
       
        # Play audio
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
       
        # Wait for playback to complete
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
       
        time.sleep(0.3)  # Small pause after speaking
       
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        print(f"   Text was: {clean_text}")



# --------------------------------------------------
# SPEECH-TO-TEXT (GOOGLE SPEECH RECOGNITION)
# --------------------------------------------------


def listen_for_speech():
    """
    Listen for user speech using Google's Speech Recognition
    Much better accuracy for Hindi than Vosk
    Returns: (text, success)
    """
    recognizer = sr.Recognizer()
   
    print("\n🎙️ आप बोलिए...")
   
    with sr.Microphone() as source:
        # Adjust for ambient noise
        print("🔧 Adjusting for background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
       
        try:
            print("👂 Listening...")
           
            # Listen for audio
            audio = recognizer.listen(
                source,
                timeout=LISTENING_TIMEOUT,
                phrase_time_limit=PHRASE_TIME_LIMIT
            )
           
            print("🔄 Processing...")
           
            # Recognize speech using Google's API
            text = recognizer.recognize_google(audio, language="hi-IN")
           
            print(f"📝 Recognized: {text}")
            return text, True
           
        except sr.WaitTimeoutError:
            print("⏱️ Timeout - no speech detected")
            return "", False
           
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            return "", False
           
        except sr.RequestError as e:
            print(f"❌ Google API Error: {e}")
            print("⚠️ Check your internet connection")
            return "", False
           
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return "", False



# --------------------------------------------------
# LLM SETUP (OLLAMA)
# --------------------------------------------------


print("⏳ Setting up LLM...")


llm = OllamaLLM(
    model="llama3.2:3b",
    temperature=0.7,
    num_predict=100  # Keep responses short
)


memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="user_input",
    return_messages=False
)


prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template=f"""{SYSTEM_PROMPT}


Previous conversation:
{{chat_history}}


User said: {{user_input}}


Prachi's response (remember: add [702] or [701] at end):"""
)


chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False
)


print("✅ LLM ready")



# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------


def extract_status(text):
    """Extract [701] or [702] from response"""
    if "[701]" in text:
        return 701
    if "[702]" in text:
        return 702
    return 702  # Default: continue



def clean_response(text):
    """Remove status codes for TTS"""
    cleaned = re.sub(r"\[70[12]\]", "", text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned



def add_status_if_missing(text):
    """Ensure response has status code"""
    if "[701]" not in text and "[702]" not in text:
        return f"{text} [702]"
    return text



# --------------------------------------------------
# MAIN VOICE CONVERSATION
# --------------------------------------------------


def start_voice_conversation():
    """
    Main voice-to-voice conversation loop
    """
    print("\n" + "="*60)
    print("📞 KOVON VOICE CHATBOT - VOICE-TO-VOICE MODE")
    print("="*60)
    print("✓ Speak naturally in Hindi")
    print("✓ Wait for Prachi to finish speaking")
    print("✓ Say 'बंद करो' or 'नहीं चाहिए' to end call")
    print("✓ Internet connection required for speech services")
    print("="*60 + "\n")
   
    # Initial greeting
    speak("नमस्ते! मैं Kovon से Prachi बोल रही हूँ।")
    time.sleep(0.5)
    speak("Kovon आपको verified agencies से safely connect करता है।")
    time.sleep(0.5)
    speak("क्या आपको overseas jobs में interest है?")
   
    conversation_count = 0
    max_conversations = 20
    no_speech_count = 0
   
    while conversation_count < max_conversations:
        conversation_count += 1
       
        print(f"\n{'─'*60}")
        print(f"Turn {conversation_count}")
        print(f"{'─'*60}")
       
        # LISTEN TO USER
        user_text, success = listen_for_speech()
       
        if not success or not user_text:
            no_speech_count += 1
            print(f"👤 User: [No speech detected]")
           
            if no_speech_count >= 2:
                speak("मुझे आपकी आवाज़ नहीं आ रही। कृपया बाद में call करें।")
                break
           
            speak("क्या आप सुन पा रहे हैं? कृपया बोलिए।")
            continue
       
        no_speech_count = 0  # Reset counter
        print(f"👤 User: {user_text}")
       
        # Check for end conversation keywords
        end_keywords = ['बंद', 'रोको', 'नहीं चाहिए', 'interest नहीं', 'रुको', 'bye', 'goodbye', 'stop']
        if any(keyword in user_text.lower() for keyword in end_keywords):
            speak("ठीक है, कोई बात नहीं। धन्यवाद! नमस्ते!")
            break
       
        # GET LLM RESPONSE
        try:
            print("💭 Thinking...")
            response = chain.invoke({"user_input": user_text})
           
            if isinstance(response, dict):
                llm_output = response.get('text', '')
            else:
                llm_output = str(response)
           
            # Ensure status code
            llm_output = add_status_if_missing(llm_output)
           
            # Extract status
            status_code = extract_status(llm_output)
           
            # Clean text for speaking
            speech_text = clean_response(llm_output)
           
            # SPEAK RESPONSE
            if speech_text:
                speak(speech_text)
           
            # Check if conversation should end
            if status_code == 701:
                print("\n✓ Call completed successfully [701]")
                break
       
        except Exception as e:
            print(f"❌ Error: {e}")
            speak("माफ़ कीजिए, कुछ technical problem हो गई। कृपया बाद में try करें।")
            break
       
        time.sleep(0.5)  # Brief pause between turns
   
    if conversation_count >= max_conversations:
        speak("समय की कमी है। हमारी team आपको contact करेगी। धन्यवाद!")
   
    print("\n" + "="*60)
    print("📞 Call Ended")
    print("="*60 + "\n")



# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------


if __name__ == "__main__":
 
    # Test pygame
    try:
        print("🔊 Testing audio system...")
        test_tts = gTTS(text="टेस्ट", lang='hi')
        fp = BytesIO()
        test_tts.write_to_fp(fp)
        fp.seek(0)
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        print("✅ Audio system working")
    except Exception as e:
        print(f"❌ Audio test failed: {e}")
        exit(1)
   
    input("\nPress ENTER to start the call...")
   
    try:
        start_voice_conversation()
    except KeyboardInterrupt:
        print("\n\n⚠️ Call interrupted by user (Ctrl+C)")
        speak("Call disconnect हो गई। धन्यवाद!")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Shutting down gracefully...")
        pygame.mixer.quit()
        time.sleep(1)