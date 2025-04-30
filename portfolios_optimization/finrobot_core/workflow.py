from typing import Any, Callable, Dict, List, Optional, Annotated
import autogen
from autogen.cache import Cache
from autogen import (
    ConversableAgent,
    AssistantAgent,
    UserProxyAgent,
    register_function,
)
from abc import ABC, abstractmethod
from .utils import *
from .prompts import leader_system_message, role_system_message


class FinRobot(AssistantAgent):

    def __init__(
        self,
        agent_config: str | Dict[str, Any],
        system_message: str | None = None,  # overwrites previous config
        toolkits: List[Callable | dict | type] = [],  # Kept for signature compatibility, but unused
        proxy: UserProxyAgent | None = None,
        **kwargs,
    ):
        orig_name = ""
        if isinstance(agent_config, str):
            # If only a name string is provided, create a basic config
            orig_name = agent_config
            agent_config = {"name": orig_name, "profile": system_message or "", "description": orig_name}

        # Ensure agent_config is a dict now
        if not isinstance(agent_config, dict):
             raise ValueError("agent_config must resolve to a dictionary.")

        # Use provided system_message if available, otherwise fallback to profile in config
        config_profile = self._preprocess_config(agent_config.copy()).get('profile', '') # Preprocess returns modified config
        final_system_message = system_message if system_message is not None else config_profile

        assert agent_config.get("name", ""), f"name needs to be in config."
        name = agent_config["name"]
        description = agent_config.get("description", name)

        # Toolkits are ignored in this simplified version
        self.toolkits = []

        name = name.replace(" ", "_").strip()

        # Pass final system message and description
        super().__init__(name, final_system_message, description=description, **kwargs)

        # Register proxy if provided (needed for user_proxy interaction)
        if proxy is not None:
            self.register_proxy(proxy)

    def _preprocess_config(self, config):
        # This function mainly formats the profile/system_message based on optional fields
        # We keep it as it's called by __init__
        role_prompt, leader_prompt, responsibilities = "", "", ""

        if "responsibilities" in config:
            title = config.get("title", config.get("name", ""))
            if "name" not in config: config["name"] = title
            responsibilities = config["responsibilities"]
            responsibilities = (
                "\n".join([f" - {r}" for r in responsibilities])
                if isinstance(responsibilities, list)
                else responsibilities
            )
            # Use role_system_message from prompts.py
            role_prompt = role_system_message.format(
                title=title,
                responsibilities=responsibilities,
            )

        name = config.get("name", "")
        description = (
            f"Name: {name}\nResponsibility:\n{responsibilities}"
            if responsibilities
            else f"Name: {name}"
        )
        config["description"] = description.strip() # Set description in config

        if "group_desc" in config:
            group_desc = config["group_desc"]
            # Use leader_system_message from prompts.py
            leader_prompt = leader_system_message.format(group_desc=group_desc)

        # Combine prompts into the profile field of the config
        config["profile"] = (
            (role_prompt + "\n\n").strip()
            + (leader_prompt + "\n\n").strip()
            + config.get("profile", "") # Append original profile if exists
        ).strip()

        return config

    def register_proxy(self, proxy):
        # register_toolkits(self.toolkits, self, proxy) # Removed dependency on toolkits
        # We might still need to register basic autogen functions if proxy needs them
        # For now, assume UserProxyAgent handles its own function registration needs
        pass


class SingleAssistantBase(ABC): # Abstract base class for SingleAssistant

    def __init__(
        self,
        # agent_config: str | Dict[str, Any], # Handled in subclass
        # llm_config: Dict[str, Any] = {}, # Handled in subclass
        assistant_instance: FinRobot # Expect an initialized FinRobot instance
    ):
         self.assistant = assistant_instance

    @abstractmethod
    def chat(self, message: str, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass


class SingleAssistant(SingleAssistantBase):

    # Modified __init__ to directly accept name and system_message
    def __init__(
        self,
        name: str, # Use name directly
        system_message: str, # Use system_message directly
        llm_config: Dict[str, Any] = {},
        # More robust termination message check
        is_termination_msg = lambda msg: isinstance(msg, dict) and str(msg.get("content", "")).rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,
        },
        **kwargs,
    ):
        # Create the underlying FinRobot (AssistantAgent)
        assistant_robot = FinRobot(
            agent_config=name, # Pass name, FinRobot handles basic config creation
            system_message=system_message, # Pass system message directly
            llm_config=llm_config,
            **kwargs # Pass other AssistantAgent kwargs
        )

        # Call parent __init__ with the created assistant instance
        super().__init__(assistant_instance=assistant_robot)

        # Create and configure the UserProxyAgent
        self.user_proxy = UserProxyAgent(
            name="User_Proxy",
            is_termination_msg=is_termination_msg,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=0,
            code_execution_config=code_execution_config,
        )
        # Register the proxy with the assistant (FinRobot handles this)
        self.assistant.register_proxy(self.user_proxy)

    def chat(self, message: str, use_cache=False, **kwargs):
        """Initiates a chat with the assistant and returns the final response content."""
        chat_res = None
        final_response_content = None
        # Remove the Cache context manager or force use_cache=False
        # with Cache.disk(cache_seed=41) as cache: # Use a fixed seed for caching if desired
        # Initiate chat. UserProxyAgent sends the message to AssistantAgent (FinRobot)
        chat_res = self.user_proxy.initiate_chat(
            self.assistant, # The recipient is the FinRobot assistant
            message=message,
            # cache=cache if use_cache else None, # Pass None for cache explicitly
            cache=None, # Force cache to be None
            **kwargs,
        )

        # Extract the last message content after chat completion
        if chat_res and chat_res.chat_history:
            # The last message in the history is usually the assistant's final reply
            last_message = chat_res.chat_history[-1]
            final_response_content = last_message.get('content')
        elif chat_res and hasattr(chat_res, 'summary'):
             # Sometimes a summary is available if history isn't directly populated
             final_response_content = chat_res.summary
        else:
             # As a fallback, check the assistant's internal message history
             try:
                 if self.assistant.chat_messages and self.user_proxy in self.assistant.chat_messages:
                    last_msg = self.assistant.chat_messages[self.user_proxy][-1]
                    final_response_content = last_msg.get('content')
             except Exception:
                  pass # Ignore errors trying the fallback

        # Reset agents after chat? Optional, might clear necessary state for follow-ups.
        # self.reset()

        return final_response_content if final_response_content is not None else "Chat finished, but no specific response content could be extracted."

    def reset(self):
        """Resets both the user proxy and the assistant agent."""
        # print("Resetting User Proxy and Assistant...") # Debug print
        self.user_proxy.reset()
        self.assistant.reset()

# --- Removed SingleAssistantRAG, SingleAssistantShadow, MultiAssistantBase, etc. --- 
# --- as they are not needed for the current integration ---
