import uuid

from langfuse import get_client, observe
from pydantic_ai import Agent

from .model import RouterFailure, VerificationOutput


class VerifyingAgent:
    agent: Agent
    verifier: Agent
    max_retries: int

    def __init__(
        self,
        agent_prompt: str,
        verifier_prompt: str,
        model,
        agent_deps,
        struct_out_agent,
        max_retries=3,
    ):
        self.agent = Agent(
            model=model,
            deps_type=agent_deps,
            system_prompt=agent_prompt,
            output_type=struct_out_agent,
            instrument=True,
        )
        self.verifier = Agent(
            model=model,
            system_prompt=verifier_prompt,
            output_type=VerificationOutput,
            instrument=True,
        )
        self.max_retries = max_retries

    @observe()
    async def run(self, message, deps):
        langfuse = get_client()
        langfuse.update_current_trace(session_id=f"{uuid.uuid4()}")
        for _attempt in range(self.max_retries):
            result = await self.agent.run(message, deps=deps)
            # Get class name of the output (e.g., NounCard, VerbCard, RouterFailure)
            output_class = type(result.output).__name__
            str_output = f"{output_class} | {str(result.output)}"
            approval_result = await self.verifier.run(str_output)
            if isinstance(approval_result.output, VerificationOutput):
                verification_output = approval_result.output
                if verification_output.approved or verification_output.uncertain:
                    return result
                message += f"\nVerifier feedback: {verification_output.reason}"
            else:
                raise Exception(
                    f"Verification agent returned invalid output -> {approval_result.output}"
                )
        result.output = RouterFailure(
            explanation=f"Verification failed after {self.max_retries} retries"
        )
        return result
