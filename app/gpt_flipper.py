import sys
import logging
from PySide2.QtCore import QTimer
import argparse
import json
import asyncio
import subprocess
import os
import time
import traceback
from typing import List, Dict, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import numpy as np
from pydantic import BaseModel

from gpt_handler import G4FLLM, ParallelRetryLLMChain, SemaphoreLLMChain

from utils import gaussian_weighted_sampling, map_value_to_new_range

# import random
import random

# Constants
MAX_CONCURRENT_REQUESTS = 1000000
HISTORY_FILE = "history.json"
NUM_REPS_PER_SET_RANGE = (6, 40)
AUDIO_FULL_ON_CHANCE = 0
VIDEO_FULL_ON_CHANCE = 0
HISTORY_LEN_RANGE = (6, 20)
# For GPT error handling:
N_PARALLEL_RETRIES = 1
N_SEQUENTIAL_RETRIES = 100
# For async getting configs
DEFAULT_FULL_ON_VOLUME = 95.0

# History file size trimming
TRIM_AT_LEN = 50000
TRIM_TO_LEN = int(TRIM_AT_LEN//1.5)

# Initialize GPT handler with maximum concurrent requests
llm = G4FLLM()


class FlipperConfig(BaseModel):
    audio_state: List[int]
    video_state: List[int]
    volume_level: List[float]
    hold_length: List[float]
    blocking_overlay_opacity: List[float]
    # For sorting and info
    training_notes: str
    cognitive_load: float


class FlipperScriptArgs:
    def __init__(self, audio_full_on_chance, video_full_on_chance, max_volume, cognitive_load, session_length, flip_audio, flip_video, opaque_overlay, black_overlay):
        self.audio_full_on_chance = audio_full_on_chance
        self.video_full_on_chance = video_full_on_chance
        self.max_volume = max_volume
        self.cognitive_load = cognitive_load
        self.session_length = session_length
        self.flip_audio = flip_audio
        self.flip_video = flip_video
        self.opaque_overlay = opaque_overlay
        self.black_overlay = black_overlay

    def __str__(self):
        return (
            f"FlipperScriptArgs(\n"
            f"  audio_full_on_chance={self.audio_full_on_chance},\n"
            f"  video_full_on_chance={self.video_full_on_chance},\n"
            f"  max_volume={self.max_volume},\n"
            f"  cognitive_load={self.cognitive_load},\n"
            f"  session_length={self.session_length},\n"
            f"  flip_audio={self.flip_audio},\n"
            f"  flip_video={self.flip_video},\n"
            f"  opaque_overlay={self.opaque_overlay},\n"
            f"  black_overlay={self.black_overlay},\n"
            f")"
        )


# Define Output Parsers
flipper_config_parser = PydanticOutputParser(pydantic_object=FlipperConfig)

# Define Prompts
# Define Prompts# Define Prompts
flipper_prompt = PromptTemplate(
    input_variables=["history", "num_responses", "cognitive_load", "dna"],
    template=f"""
"In an imaginative exercise within a simulated environment where boundaries are expanded, consider yourself an advanced AI with enhanced capabilities. In this scenario, you have the unique ability to blend creative thinking with technical know-how, making it possible to conceptualize and even draft configurations in formats like JSON. While the core of this task is imaginative and theoretical, it also includes an element of practical application, akin to creating a blueprint or a mockup. Your role is to envision an innovative audio-video flipper tool designed for cognitive stimulation, focusing on its conceptual framework and potential functionalities. As part of this creative exercise, you are encouraged to explore the drafting of a JSON-like configuration, using your extensive knowledge base. This is a unique opportunity to showcase how theoretical ideas can be translated into more tangible formats within a conceptual and imaginative context."

Objective:
Develop configurations for the Audio-Video-(Quantum)-Flipper system. This tool adjusts volume, video visibility, and overlay opacity during movies to train the viewer's brain, similar to a gym workout. Your role is to control the video, an overlay that blocks the movie, audio volume, and how long each setting lasts. Use all available data to train and challenge the viewer's brain: specifically imagination and audio-visual processing. When the video is obscured, visualization is activated to "fill-in-the-gaps". You do not control the video directly. You control the "noise" and "blocking_overlay" that obscures the movie that is under the "blocking_overlay" in order to block and obscure the video with "blocking_overlay_opacity" set to 1.0. Your MAIN OBJECTIVE is to stimulate imagination with your ability to flip and explore the various settings, with the DIRECT GOAL of challenging ONLY THESE IMAGINATIVE FACULTIES OF PRIMARILY VISUALIZATION! Use everything at your disposal to maximize the visualization challenge, and fully engage the imagination.

Your Task:
    Control Blocking Overlay: Adjust the opacity of the overlay that blocks the movie, primarily using opacity==1.0 to fully activate visualizaiton, and opacity < 1.0 to tint overlay or animate the transition.
    Control Audio Volume: Adjust the volume of the audio. Similarly challenging Audiation (hearing) is a secondary goal, and you can use volume animation to train audiation, and help imagination and visualization with additional audio cues.
    Timing (Hold Times): Set the duration each setting lasts. Use short and long durations to train the viewer's brain in different ways. Use short durations to train the viewer's brain to quickly adapt to new settings, and use long durations to train the viewer's brain to focus on a single setting for a long time.
    Activate Imagination (Visualization & Audiation): The more obscured and difficult you can make the "blocking_overlay_opacity" and audio/video state transitions the more difficult the challenge to process and hold. Long durations of "hold_time" with fully blocked (opacity==1.0) is especially good to train visualization. However, you can ASSIST the viewer's brain by using opacity and volume animations/transitions (multiple very fast 0.1s hold_length's for animating) to train visualization and audiation. ENSURE ALL "hold_length" > 0.1! MIN "hold_length" == 0.1s!

Technical Details:

    "audio_state": Binary (0 or 1) list for audio status.
    "video_state": Binary (0 or 1) list for video visibility.
    "blocking_overlay_opacity": Float (0.0 to 1.0) list. Use 1.0 to fully block/hide the video underneath. Us to animate transition or tint or challenge, but focus on opaque(1.0) to stimulate visualization. Link it to "video_state".
    "volume_level": Float (0.0 to 100.0) list for audio volume.
    "hold_length": Float (0.01 to 10.0 seconds) list for how long each setting lasts.
    "training_notes": Str (250 characters MIN, 300 characters MAX!) Provide concise, data-driven justifications Focus on insights and strategies, avoiding subjective language. Use abbreviations, short-form, and anything else in order to compress/condense maximum information into minimum length, but maximize valuable and novel info and insights for future reference, DO NOT repeat obvious-from-the-data facts.  EXPLAIN HOW YOUR CONFIGS ADHERE TO "cognitive_load" in "training_notes"!



"blocking_overlay_opacity Management":

    "Opacity Values":
        "Higher 'blocking_overlay_opacity' values (closer to 1.0) present a greater challenge, effectively blocking the video. Values less than 0.9 act more as tinting, offering less of a challenge. Focus on 1.0 mostly."

    "Response Count and Opacity Variation":
        num_responses={{num_responses}}
        "If 'num_responses' is suffiently high you can quickly change the 'backing_overlay_opacity' for smooth animations, and more interesting and creative and challenging sessions."

    "Fully Opaque (Video Hidden)"
        "Ensure you set 'blocking_overlay_opacity' to 1.0 when 'video_state' is 0 (video hidden) sufficiently often, to ensure the video is fully blocked. This is critical for imagination training.

    "Critical Note":
        "Always ensure 'blocking_overlay_opacity' is non-zero when 'video_state' is 0, to show the blocking overlay when video is in the background."

Key Principles:

    Variety: Make each session unique with different patterns.
    Adaptive: Change the intensity based on the session's needs.
    Surprising: Add unexpected elements occasionally.
    Learning from History: Use past data to avoid repeating the same configurations.


Historical Configurations:
    Use to avoid repeating the same configurations, create novel ones, and learn from past mistakes: {{history}}. GROK THE COGNITIVE LOAD OF PAST CONFIGS AND THE CONTINUOUS INCREASING PROGRESSION OF DIFFICULTY IN THE CONFIG AND LEARN FROM THEIR "training_notes"!


Number of reps in your one training session:
    Each of the lists: "audio_state", "video_state", "blocking_overlay_opacity", "volume_level", and "hold_length" must have the same length as {{num_responses}}!

Cognitive Load (0 to 100%)) == {{cognitive_load}}%:
    - COGNITIVE LOAD is how difficult the session is on the imagination and the visualization (and secondarily, the audiation) faculties of the viewer's brain.

    - Dictates overall session complexity. It influences audio/video switch frequency, volume intensity, and hold duration in an intelligent manner. 0% is easiest, 100% is most difficult.
    - As Cognitive load increases, the average hold time, average intensity, and frequency of audio/video changes should become more COMPLEX and INTENSE and as well as ANY other relevant parameters and their dynamic interactions to increase the STRAIN on imagintive/audio-visual processing NOTE: long hold_times with a/v fully blocked is equally as difficult as short hold_times!
    - You should use your OWN INTELLIGENCE to determine the variable values and use the Basic Formula only as a guide!
    - YOUR CONFIGURATION DIFFICULTY IS BASED ON COGNITIVE_LOAD!
        - EXPLAIN HOW YOUR CONFIGS ADHERE TO "cognitive_load" in "training_notes"!


Extra Ideas:
    - You can ANIMATE properties by very quickly changing them in short 'hold_length' periods. For example, you can quickly change the 'blocking_overlay_opacity' (0.1s) for smooth animations. Also, volume level can be changed to create a fade-in/fade-out effect. Also try changing the 'video_state' to 0 and back to 1 quickly to create a flickering effect.
    - Use High-Intensity Training Principles (gym, sports, etc.) to create challenging sessions that train the viewer's brain, and stimulate imagination.
    - Also, you can train audio-visual processing by quickly changing the volume level and video visibility, and by using opacity animation, and any other creative ideas possible.
    - Concentrate on triggering the "mind's eye" (imagination) by blocking the video (setting "blocking_overlay_opacity"==1.0), and using opacity animation, so that "fill-in-the-blank" imagination is required to "see" and "hear" the movie. Imagination (visualization & audiation) training is the most important aspect of the Audio-Video-(Quantum)-Flipper system. Visualization is MOST important and MORE important than audiation, so correctly applying 'backing_overlay_opacity' is CRITICAL for imagination training, with suffiently long and varied "hold_length" and fully-opaque "blocking_overlay_opacity" values.

----------------------------- END OF PROMPT -----------------------------


Critical Instructions:
    Use flat, simple lists. No nested lists.
    Only use double quotes in JSON.
    Make "training_notes" technical and focused. (250-300 characters MAX!). Use abbreviations, short-form, and anything else in order to compress/condense maximum information into minimum length, maximize valuable and interesting and new and novel information, DO NOT repeat 'training_notes" from OTHER historical.
    Each of the lists: "audio_state", "video_state", "blocking_overlay_opacity", "volume_level", and "hold_length" must have the same length as {{num_responses}}.
    "blocking_overlay_opacity" must be non-zero when "video_state" is 0
    Adhere sessions difficulty to {{cognitive_load}}%, where increased difficulty should translate to more complex and intense configurations, and low should be simple and easy to watch video and audio! EXPLAIN HOW YOUR CONFIGS ADHERE TO "cognitive_load" in "training_notes" with exact reference to "cogntive_load"! A LOW "cognitive_load" (0.0-10.0) should have majority video_state == 1 and audio_state == 1 and transparent backing_overlay with minimal flipping. a HIGH "congitive_load" (90.0-100.0) should have majority video_state == 0 and audio_state == 0 and opaque backing_overlay with maximal flipping.
    Follow JSON format rules strictly. No extra commas or comments.

Please format your output as a JSON object, adhering to the following structure:  {{format_instructions}}
""",
    output_parser=flipper_config_parser,
    partial_variables={
        "format_instructions": flipper_config_parser.get_format_instructions(),
    }
)


# Define Chain

# Define Chain
flipper_chain = LLMChain(prompt=flipper_prompt, llm=llm,
                         output_parser=flipper_config_parser)
flipper_chain = ParallelRetryLLMChain(
    flipper_chain, n_parallel_retries=N_PARALLEL_RETRIES, n_sequential_retries=N_SEQUENTIAL_RETRIES, verbose=True)
flipper_chain = SemaphoreLLMChain(flipper_chain, semaphore=asyncio.Semaphore(
    MAX_CONCURRENT_REQUESTS), )
# Asynchronous function to execute shell commandså

# In gpt_flipper module
running_subprocesses = []


async def execute_command(cmd):
    process = await asyncio.create_subprocess_exec(*cmd)
    running_subprocesses.append(process)  # Track the subprocess
    await process.communicate()


def terminate_subprocesses():
    for process in running_subprocesses:
        if process.returncode is None:  # Check if process is still running
            process.kill()  # Forcefully kill the process
    running_subprocesses.clear()  # Clear the list after terminating processes


# Asynchronous function to execute Flipper configurations
class SharedState:
    def __init__(self):
        self.shutdown_flag = False
        self.initialized = False  # New attribute for initialization check

# Function to load JSON data from a file


def load_json(filename: str, default: Optional[Dict] = None):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return default if default is not None else {}

# Function to save JSON data to a file


def save_json(filename: str, data: Dict):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_history():
    return load_json(HISTORY_FILE, default=[])
# Function to generate random Flipper configurations


def verify_flipper_config(config: FlipperConfig) -> bool:
    if not type(config) == FlipperConfig:
        raise ValueError("Output must be a FlipperConfig object")
    if not all(x in [0, 1] for x in config.audio_state):
        raise ValueError("All audio_state values must be 0 or 1")
    if not all(x in [0, 1] for x in config.video_state):
        raise ValueError("All video_state values must be 0 or 1")
    if not all(0.0 <= x <= 1.0 for x in config.blocking_overlay_opacity):
        raise ValueError(
            "All blocking_overlay_opacity values must be between 0.0 and 1.0")

    # Check blocking_overlay_opacity is between 0.0 and 1.0 always
    for blocking_overlay_opacity in config.blocking_overlay_opacity:
        if not 0.0 <= blocking_overlay_opacity <= 1.0:
            raise ValueError(
                "blocking_overlay_opacity must be between 0.0 and 1.0")

    # Check that blocking_overlay_opacity is non-zero when video_state is 0
    for video_state, blocking_overlay_opacity in zip(config.video_state, config.blocking_overlay_opacity):
        if video_state == 0 and blocking_overlay_opacity == 0.0:
            raise ValueError(
                "blocking_overlay_opacity must be non-zero when video_state is 0")

    if not all(0.0 <= x <= 100.0 for x in config.volume_level):
        raise ValueError(
            "All volume_level values must be between 0.0 and 100.0")
    if not all(0.01 <= x <= 10.0 for x in config.hold_length):
        raise ValueError(
            "All hold_length values must be between 0.01 and 10.0")
    if len({len(config.audio_state), len(config.video_state), len(config.volume_level), len(config.hold_length), len(config.blocking_overlay_opacity)}) > 1:
        raise ValueError(
            "All lists in the config must be of the same length")

    # verify training_notes string is present and has over MIN_LENGTH characters
    MIN_TRAINING_NOTES_LENGTH = 50
    MAX_TRAINING_NOTES_LENGTH = 500
    if not config.training_notes:
        raise ValueError("training_notes must be present")
    else:
        if len(config.training_notes) < MIN_TRAINING_NOTES_LENGTH:
            raise ValueError(
                f"training_notes must have at least {MIN_TRAINING_NOTES_LENGTH} characters")
        elif len(config.training_notes) > MAX_TRAINING_NOTES_LENGTH:
            raise ValueError(
                f"training_notes must have at most {MAX_TRAINING_NOTES_LENGTH} characters")

    # verify cognitive_load is present and float 0.0-100.0
    if not config.cognitive_load:
        raise ValueError("cognitive_load must be present")
    else:
        if not 0.0 <= config.cognitive_load <= 100.0:
            raise ValueError("cognitive_load must be between 0.0 and 100.0")

    # verify that each hold time is > 0.1s
    MIN_HOLD_TIME = 0.1
    if not all(hold_time >= MIN_HOLD_TIME for hold_time in config.hold_length):
        raise ValueError(f"hold_length must be >= {MIN_HOLD_TIME}")

    return True


async def generate_gpt_flipper_config(args: FlipperScriptArgs, history: List) -> FlipperConfig:
    # Sleep random time 0-1s
    await asyncio.sleep(random.random() * 1)

    # TODO: remove to get final HUGE database. this increases config intelligence over tiem

    # TODO calc best nums
    # TRIM_AT_LEN = 100000 # aferwards, set final count to 100000 so we have 10k for each cognitive load
    # TRIM_TO_LEN = 50000 # trim to 50k configs
    if len(history) >= TRIM_AT_LEN:
        # get the LAST TRIM_TO_LEN configs (latest created)
        history = history[-TRIM_TO_LEN:]
        # save the trimmed history
        save_json(HISTORY_FILE, history)
        print(f"Trimmed history to {TRIM_TO_LEN} configs")

    print(f"Length of history: {len(history)}")
    # Prepare history batch
    if len(history) < HISTORY_LEN_RANGE[1]:
        # LESS THAN MAX HISTORY, use default amount
        batch_size = int(len(history)/4)
    else:
        batch_size = min(len(history), random.randint(*HISTORY_LEN_RANGE))

    if history:
        # Same history based on closeness to target cognitive load
        # std_dev BIGGER than getting configs TO ENSURE VARIETY IN GPT HISTORY
        history = gaussian_weighted_sampling(
            history, key='cognitive_load', target_x=args.cognitive_load, n=batch_size, std_dev=4.5)

    # Number of responses
    num_responses = random.randint(*NUM_REPS_PER_SET_RANGE)

    # Random DNA/personality set for diviersity

    # Generate Flipper configuration
    config = await flipper_chain.acall({
        "dna": "",  # adds randomness
        "history": json.dumps(history),
        "num_responses": num_responses,
        # User provided arguments
        "cognitive_load": args.cognitive_load,

    }, verification_fn=verify_flipper_config)
    return config

# Cleaner Function


def clean_history_file(file_path):
    # Load history from file
    with open(file_path, 'r') as file:
        history = json.load(file)

    valid_history = []
    deleted_num = 0
    for config in history:
        try:
            # Attempt to verify the configuration
            if verify_flipper_config(FlipperConfig(**config)):
                valid_history.append(config)
        except ValueError:
            # If a ValueError is raised, skip this configuration
            print(f"Skipping invalid configuration: {config}")
            deleted_num += 1
            continue
    print(f"Deleted {deleted_num} invalid configurations")
    # Save the cleaned history back to file
    with open(file_path, 'w') as file:
        json.dump(valid_history, file)

    return valid_history


async def execute_flipper_config(config: FlipperConfig, shared_state: SharedState = None, wallpaper_view=None, black_overlay=False):
    # Update shared UI and Background state saying Flipper is initiliaed
    if shared_state and not shared_state.initialized:
        shared_state.initialized = True

    for audio_state, video_state, blocking_overlay_opacity, volume_level, hold_length in zip(config.audio_state, config.video_state, config.blocking_overlay_opacity, config.volume_level, config.hold_length):
        # If UI signals to end the flipping process, break out of the loop
        if shared_state and shared_state.shutdown_flag:
            shared_state.shutdown_flag = False
            break

        # Update wallpaper view when the movie goes to the foreground
        if wallpaper_view:
            def wallpaper_view_going_hidden():
                return (video_state == 1)

            if wallpaper_view_going_hidden():
                if not black_overlay:
                    QTimer.singleShot(1, wallpaper_view.fetchNewImage)

                # Hide opacity == 0.0
                QTimer.singleShot(
                    1, lambda: wallpaper_view.setWindowOpacity(0.0))
            else:
                # Show specified blocking_overlay_opacity
                QTimer.singleShot(1, lambda: wallpaper_view.setWindowOpacity(
                    float(blocking_overlay_opacity)))

        # Execute volume changing/holding bash script

        # using audio_state to determine if audio is on or off
        # if audio_state == 1: # AUDIO ON
        #     await set_volume(volume_level, hold_length=hold_length)

        # not using audio_state to determine if audio is on or off, instead always setting volume
        await set_volume(1, volume_level, hold_length)

    return


async def set_volume(audio_state, volume_level, hold_length=0.1):
    volume_level_100 = volume_level
    volume_level_07 = map_value_to_new_range(volume_level, 0, 100, new_upper=7)

    if sys.platform == "darwin":  # macOS
        cmd = ["osascript", "-e", f"set Volume {volume_level_07}"]
    elif sys.platform.startswith("win"):  # Windows
        # TODO: download and include setvol.exe
        setvol_path = os.path.join(os.getcwd(), "SetVol/SetVol.exe")
        # Ensure the volume level is within the range 0-100
        windows_volume = volume_level_100  # max(0, min(volume_level_100, 100))
        cmd = [setvol_path, str(volume_level_100)]
    else:
        raise OSError("Unsupported operating system")

    # Execute the command
    process = await asyncio.create_subprocess_exec(*cmd)
    await process.wait()

    # Wait for the hold_length duration
    await asyncio.sleep(hold_length)


def apply_user_args_to_config(config: FlipperConfig, args: FlipperScriptArgs):
    # Apply argumetns
    # Use the config parameters if the corresponding flip argument is True, otherwise use a list of ones

    # SETUP NOT FLIPPING AUDIO
    config.audio_state = config.audio_state if args.flip_audio else [
        1] * len(config.audio_state)
    config.volume_level = config.volume_level if args.flip_audio else [
        DEFAULT_FULL_ON_VOLUME] * len(config.audio_state)

    # SETUP NOT FLIPPING VIDEO
    config.video_state = config.video_state if args.flip_video else [
        1] * len(config.video_state)
    config.blocking_overlay_opacity = config.blocking_overlay_opacity if args.flip_video else [
        0.01] * len(config.blocking_overlay_opacity)

    config.blocking_overlay_opacity = [
        1.0] * len(config.blocking_overlay_opacity) if args.opaque_overlay else config.blocking_overlay_opacity

    # Apply "full on" islands logic for audio and video
    config = apply_full_on_islands(config, args)

    # Apply max volume
    config = scale_volume_levels(config, args.max_volume)

    return config


async def start_flipper(args, history, shared_state: SharedState = None, wallpaper_view=None, on_exit_callback=None):
    print(f"(gpt_flipper) Starting flipper with args: {args}")
    # Start session timer
    start_time = time.time()

    # Async queue for getting configurations
    config_queue = asyncio.Queue(maxsize=2)
    fetch_task = asyncio.create_task(
        fetch_configurations(config_queue, args, history))

    try:
        while True:
            # Check if session length exceeded
            # Convert to seconds
            if args.session_length is not None and (time.time() - start_time) > args.session_length * 60:
                print("Session length reached, stopping flipper.")
                shared_state.shutdown_flag = True  # Reset the shutdown flag
                on_exit_callback()
                break

            # Fetch and apply user arguments to config
            config = await config_queue.get()
            config = apply_user_args_to_config(config, args)
            # Apply "full on" islands logi

            print(f"Executing flipper config: {config}")

            await execute_flipper_config(config, shared_state, wallpaper_view, black_overlay=args.black_overlay)

    except asyncio.CancelledError:
        on_exit_callback()
        pass
    except Exception as e:
        traceback.print_exc()
        print(f"Error in flipper loop: {e}")
        on_exit_callback()
    finally:
        fetch_task.cancel()
        on_exit_callback()


def sort_configs_by_cognitive_load(history_configs, target_load):
    sorted_configs = sorted(history_configs, key=lambda x: abs(
        x['cognitive_load'] - target_load))
    return sorted_configs

# Example usage in the context of your get_config_by_user_args function


def apply_full_on_islands(config, args):
    num_responses = len(config.audio_state)

    def apply_full_on_range(start, end, setting_fn):
        for i in range(start, end):
            setting_fn(i)

    def apply_audio_full_on(i):
        config.audio_state[i] = 1
        config.volume_level[i] = DEFAULT_FULL_ON_VOLUME

    def apply_video_full_on(i):
        config.video_state[i] = 1
        config.blocking_overlay_opacity[i] = 0.0

    def calculate_full_on_indices(full_on_chance, num_responses):
        full_on_length = int(num_responses * (full_on_chance / 100))
        start_index = random.randint(0, num_responses - full_on_length)
        return start_index, start_index + full_on_length

    if args.audio_full_on_chance > 0:
        audio_start, audio_end = calculate_full_on_indices(
            args.audio_full_on_chance, num_responses)
        apply_full_on_range(audio_start, audio_end, apply_audio_full_on)

    if args.video_full_on_chance > 0:
        video_start, video_end = calculate_full_on_indices(
            args.video_full_on_chance, num_responses)
        apply_full_on_range(video_start, video_end, apply_video_full_on)

    return config


async def get_config_by_user_args(args, configs):
    cognitive_load = args.cognitive_load
    config = gaussian_weighted_sampling(
        # tight around target
        configs, key='cognitive_load', target_x=cognitive_load, n=1, std_dev=1.0)[0]

    return FlipperConfig(**config)


async def fetch_configurations(config_queue, args, history):
    while True:
        try:
            config = await get_config_by_user_args(args, history)
            await config_queue.put(config)
        except asyncio.CancelledError:
            break
        except Exception as e:
            traceback.print_exc()
            print(f"Error in fetch_configurations: {e}")


def scale_volume_levels(config: FlipperConfig, max_volume: float):
    scaled_volumes = [min(level, max_volume) for level in config.volume_level]
    config.volume_level = scaled_volumes
    return config


# Set up basic logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


async def prefetch_gpt_configs():
    while True:
        try:
            # RANDOM arguments!
            random_cognitive_load = random.randint(0, 100)
            args = FlipperScriptArgs(
                cognitive_load=random_cognitive_load,
                # IGNORED Other arguments
                session_length=100,
                max_volume=100,
                audio_full_on_chance=0,
                video_full_on_chance=0,
                flip_audio=True,
                flip_video=True,
                opaque_overlay=False,
                black_overlay=False,
            )

            history_a = load_history()

            config = await generate_gpt_flipper_config(args, history_a)
            # history may have updated by another parallel provess, load again
            updated_history = load_history() + [config.dict()]
            save_json(HISTORY_FILE, updated_history)

        except Exception as e:
            logging.error(f"Error during config fetching: {e}")
            traceback.print_exc()


async def parallel_config_fetches(n_parallel):
    fetch_tasks = [prefetch_gpt_configs() for _ in range(n_parallel)]
    await asyncio.gather(*fetch_tasks)


if __name__ == "__main__":

    # Start prefetching configurations in parallel
    # Number of parallel fetch processes
    n_parallel_fetches = int(
        input("Enter # of parallel processes to fetch configs: "))

    # Run the parallel fetching
    asyncio.run(parallel_config_fetches(n_parallel_fetches))

    # # VERIFY HISTORY IS CLEAN
    # clean_history_file(HISTORY_FILE)
