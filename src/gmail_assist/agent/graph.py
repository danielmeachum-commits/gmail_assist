"""The email triage graph.

    START ─► tag_me ─► route_by_kind ─┬─ personal ─► detect_login_code ─┬─ login code ─► END
                                      │                                 └─ otherwise ──► check_thread
                                      └─ bulk ─────► review_subscription ──────────────► check_thread

    check_thread ─┬─ thread already labelled ─► END (inherits it)
                  └─ new thread ──────────────► choose_model

    choose_model ─► classify ─► after_classify ─┬─ confident ──────────► decide ──────► END
       (local)         ▲                        ├─ unsure, local only ─► escalate ─┐
                       │                        └─ unsure, strong ─────► flag_review ─► END
                       └───────────────────────────────────────────────────────────┘
"""
from langgraph.graph import END, START, StateGraph

from gmail_assist.agent import nodes
from gmail_assist.agent.state import Context, EmailState


def build_graph():
    builder = StateGraph(EmailState, context_schema=Context)

    builder.add_node('tag_me', nodes.tag_me)
    builder.add_node('detect_login_code', nodes.detect_login_code)
    builder.add_node('review_subscription', nodes.review_subscription)
    builder.add_node('check_thread', nodes.check_thread)
    builder.add_node('choose_model', nodes.choose_model)
    builder.add_node('classify', nodes.classify)
    builder.add_node('escalate', nodes.escalate)
    builder.add_node('decide', nodes.decide)
    builder.add_node('flag_review', nodes.flag_review)

    builder.add_edge(START, 'tag_me')
    builder.add_conditional_edges('tag_me', nodes.route_by_kind)
    builder.add_conditional_edges('detect_login_code', nodes.after_login_check)
    builder.add_edge('review_subscription', 'check_thread')
    builder.add_conditional_edges('check_thread', nodes.after_thread_check)
    builder.add_edge('choose_model', 'classify')
    builder.add_conditional_edges('classify', nodes.after_classify)
    builder.add_edge('escalate', 'classify')
    builder.add_edge('decide', END)
    builder.add_edge('flag_review', END)

    return builder.compile()
