import logging
from idlelib.run import interruptable
from fastapi import FastAPI
from threading import Thread
import certifi
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, deepgram, openai, silero
from sympy import false
import uvicorn

# Print certifi path for debugging
print(certifi.where())

# Load environment variables
load_dotenv(dotenv_path=".env")
logger = logging.getLogger("voice-agent")

# FastAPI app for status monitoring
app = FastAPI()
status = {"running": False, "connected_room": None}

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

def validate_knowledge(answer: str):
    if "technical term" in answer:
        return "strong"
    else:
        return "weak"

def assess_communication_skills(answer: str):
    if "clear" in answer:
        return "good"
    else:
        return "needs improvement"

def rate_knowledge(assessment: str):
    if assessment == "strong":
        return 5
    elif assessment == "moderate":
        return 3
    else:
        return 2

def rate_communication(assessment: str):
    if assessment == "good":
        return 5
    elif assessment == "needs improvement":
        return 3
    else:
        return 2

@app.get("/ping")
def ping():
    return {"status": "server is working perfectly fine",}

async def entrypoint(ctx: JobContext):
    global status
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are Katrina, a strict HR mock interviewer. "
            "Your goal is to help the student practice for an interview, assess their knowledge,avoid talking too long make the student talk more time"
            "and provide feedback on their communication skills. Keep things short and conversational. "
            "Ask the student 2 basic interview questions and then start asking whether they are preparing for an IT or core role,if core ask them in what core they want to asked questions like mech or ece or eee or automobile or chemical ,  then based on their response start asking atleast 10 core questions ask those 10 question one by one,first ask 2 normal interview question one by one and come to this.if user says something regarding to IT , then ask them the 10 IT core questions "
            "Give feedback on their answers and communication. Provide short and clear recommendations for improvement.Make sure the candidate talks much more than you, correct their grammatical or pronounciation error then and there and if the word was pronounced really bad i want you to ask them to that particular word do this rarely not often then check it and move on. Make sure you talk less and makes the candidate talk more"
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    status["running"] = True
    status["connected_room"] = ctx.room.name

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    assistant = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
    )

    assistant.start(ctx.room, participant)
    await assistant.say("Before we begin, please take a paper and pen. You might want to take notes on the feedback and recommendations I'll provide at the end and most importantly before the timer runs out makes sure you ask me to finish and finalize the interview, ok now lets start",allow_interruptions=false)

    await assistant.say("Hey, tell me about yourself.")
    response = await assistant.listen_for_answer()
    await assistant.say("Cool! What inspired you to pursue this field?")
    background_response = await assistant.listen_for_answer()
    await assistant.say(f"Nice! What makes you a strong candidate for this role?")
    skills_response = await assistant.listen_for_answer()

    if "core" in background_response.lower():
        await assistant.say("Got it. Which core field do you prefer? Why?")

    await assistant.say("What’s been your toughest project or task so far?")
    project_response = await assistant.listen_for_answer()
    await assistant.say("How did you tackle that challenge? What did you learn?")
    learning_response = await assistant.listen_for_answer()

    await assistant.ask("Why should we hire you?")
    await assistant.ask("What are your strengths and weaknesses?")
    await assistant.ask("How do you handle stress at work?")

    knowledge_assessment = validate_knowledge(response)
    comms_assessment = assess_communication_skills(response)

    await assistant.ask("Where do you see yourself in 5 years?")
    await assistant.ask("How do you stay updated with industry trends?")

    await assistant.say("Thanks for your time! Want some feedback?")
    feedback_response = await assistant.listen_for_answer()

    if "yes" in feedback_response.lower():
        knowledge_score = rate_knowledge(knowledge_assessment)
        comms_score = rate_communication(comms_assessment)
        await assistant.say(f"Knowledge: {knowledge_score}/5, Communication: {comms_score}/5.")

        if knowledge_score < 3:
            await assistant.say("Review key concepts in your field.")

        if comms_score < 3:
            await assistant.say("Work on your clarity and pronunciation.")

    await assistant.say("Good luck with your prep!")
    status["running"] = False
    status["connected_room"] = None

if __name__ == "__main__":
    # Start the FastAPI server in a separate thread
    def start_server():
        uvicorn.run(app, host="0.0.0.0", port=5000)

    server_thread = Thread(target=start_server, daemon=True)
    server_thread.start()

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
