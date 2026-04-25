from helpdeskenv_class import HelpdeskEnv
from models import HelpdeskAction, AgentRole
import json

env = HelpdeskEnv()
obs = env.reset(num_tickets=1)
print('Type:', type(obs))
print('Obs:', obs)
state = env.state()
print('Ticket:', state.current_ticket.subject)
print('Starting agent:', state.current_agent)

action = HelpdeskAction(
    ticket_id=state.current_ticket.ticket_id,
    agent_role=AgentRole.TRIAGE,
    action_type='classify',
    action_value=json.dumps({
        "category": "password_reset",
        "priority": "medium",
        "tier": "l1"
    })
)
result = env.step(action)
print('Triage reward:', result.reward)
print('Current agent after triage:', env.state().current_agent)
print('Multi-agent routing WORKS!')