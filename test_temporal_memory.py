#!/usr/bin/env python3
"""
Test script for Temporal Graph Memory integration
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

def main():
    from src.text_analysis import TextAnalyzer
    from src.temporal_graph_memory import TemporalGraphMemory

    print("=" * 50)
    print("Testing Temporal Graph Memory")
    print("=" * 50)
    
    # Initialize
    print("\n1. Initializing TextAnalyzer...")
    ta = TextAnalyzer()
    print("   ✓ TextAnalyzer ready")

    print("\n2. Initializing TemporalGraphMemory...")
    tm = TemporalGraphMemory(text_analyzer=ta, storage_path='/tmp/test_memory.json')
    print("   ✓ TemporalGraphMemory ready")

    # Create a meeting
    print("\n3. Creating test meeting...")
    meeting_id = tm.create_meeting(
        title='Q2 Planning Meeting', 
        participants=['Alice', 'Bob', 'Charlie'],
        tags=['planning', 'q2', 'budget']
    )
    print(f"   ✓ Created meeting: {meeting_id[:8]}...")

    # Add segments
    print("\n4. Adding segments...")
    segments = [
        ("We need to finalize the budget for Q2 before next week.", 0.0, 5.0, "Alice", 0.9),
        ("I think we should increase marketing spend by 20%.", 5.0, 10.0, "Bob", 0.7),
        ("That's a good point. Let me check with finance.", 10.0, 15.0, "Charlie", 0.5),
        ("The product launch is scheduled for March 15th.", 15.0, 20.0, "Alice", 0.85),
    ]
    for text, start, end, speaker, score in segments:
        seg_id = tm.add_segment(meeting_id, text, start, end, speaker, score)
        print(f"   ✓ Added segment from {speaker}")

    # Add decisions
    print("\n5. Adding decisions...")
    dec1 = tm.add_decision(meeting_id, "Approved 20% increase in marketing budget", "Alice")
    dec2 = tm.add_decision(meeting_id, "Product launch confirmed for March 15th", "Alice")
    print(f"   ✓ Added 2 decisions")

    # Add action items
    print("\n6. Adding action items...")
    action1 = tm.add_action_item(meeting_id, "Update Q2 budget spreadsheet", "Bob", priority="high")
    action2 = tm.add_action_item(meeting_id, "Prepare launch marketing materials", "Charlie", priority="medium")
    action3 = tm.add_action_item(meeting_id, "Schedule follow-up meeting for next week", "Alice", priority="low")
    print(f"   ✓ Added 3 action items")

    # Add topics
    print("\n7. Adding topics...")
    topic1 = tm.add_topic(meeting_id, "Q2 Budget", "Discussion about quarterly budget allocation")
    topic2 = tm.add_topic(meeting_id, "Product Launch", "March product launch planning")
    print(f"   ✓ Added 2 topics")

    # Test context retrieval
    print("\n8. Testing semantic search...")
    queries = [
        "budget planning",
        "product launch date",
        "marketing spend"
    ]
    for query in queries:
        context = tm.get_context_for_text(query, top_k=3)
        print(f"   Query: '{query}' -> {len(context)} results")
        for item in context[:2]:
            item_type = item.get('type', 'unknown')
            text = item.get('text', item.get('name', 'N/A'))[:50]
            score = item.get('similarity', 0)
            print(f"      - [{item_type}] {text}... (score: {score:.2f})")

    # Test highlight scoring integration
    print("\n9. Testing highlight score computation...")
    test_text = "We need to finalize the marketing budget"
    if ta.embedding_model:
        emb = ta.get_embedding(test_text)
        if emb is not None:
            boost, context = tm.get_temporal_boost_for_embedding(emb)
            print(f"   ✓ Temporal boost for '{test_text[:30]}...': {boost:.3f}")
            print(f"   ✓ Retrieved {len(context)} relevant context items")

    # Save and load
    print("\n10. Testing persistence...")
    tm.save()
    print("   ✓ Saved to disk")
    
    # Create new instance and load
    tm2 = TemporalGraphMemory(text_analyzer=ta, storage_path='/tmp/test_memory.json')
    stats = tm2.get_statistics()
    print(f"   ✓ Loaded from disk: {stats}")

    # Get all meetings
    print("\n11. Listing all meetings...")
    meetings = tm2.get_all_meetings()
    for m in meetings:
        title = m.content.get('title', 'Untitled')
        print(f"   - {title} ({m.node_id[:8]}...)")

    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)

    # Cleanup
    if os.path.exists('/tmp/test_memory.json'):
        os.remove('/tmp/test_memory.json')
        print("\nCleaned up test file.")


if __name__ == "__main__":
    main()
