from __future__ import annotations

import queue
import unittest
from unittest.mock import Mock

from application.event_types import event_to_record
from asr.application.event_queue import AsrEventQueue
from asr.application.events import ASREventPublisher


class AsrEventQueueTests(unittest.TestCase):
    def test_final_evicts_disposable_update_and_survives_full_queue(self):
        events = AsrEventQueue(maxsize=1)
        publisher = ASREventPublisher(logger=Mock(), event_queue=events)
        publisher.publish({'type': 'streaming_words', 'stream': 'mic'})
        publisher.publish(
            {'type': 'streaming_final', 'stream': 'mic', 'words': ['final']}
        )

        self.assertEqual(
            event_to_record(events.get_nowait())['words'], ['final']
        )
        events.task_done()
        self.assertEqual(events.unfinished_tasks, 0)
        self.assertTrue(events.empty())

    def test_retained_events_keep_order_when_only_retained_events_are_pending(
        self,
    ):
        events = AsrEventQueue(maxsize=1)
        publisher = ASREventPublisher(logger=Mock(), event_queue=events)
        publisher.publish({'type': 'utterance', 'text': 'one'})
        publisher.publish({'type': 'streaming_final', 'words': ['two']})
        publisher.publish({'type': 'error', 'error': 'three'})
        publisher.publish({'type': 'asr_metrics'})
        records = [event_to_record(events.get_nowait()) for _ in range(3)]
        self.assertEqual(
            [record['type'] for record in records],
            ['utterance', 'streaming_final', 'error'],
        )
        self.assertTrue(events.empty())

    def test_plain_bounded_queue_does_not_silently_discard_final_text(self):
        events = queue.Queue(maxsize=1)
        events.put_nowait({'type': 'asr_metrics'})
        publisher = ASREventPublisher(logger=Mock(), event_queue=events)
        with self.assertRaises(queue.Full):
            publisher.publish({'type': 'streaming_final'})

    def test_empty_public_types_suppresses_publication_explicitly(self):
        events = AsrEventQueue()
        publisher = ASREventPublisher(
            logger=Mock(), event_queue=events, public_types=set()
        )
        publisher.publish({'type': 'streaming_final'})
        self.assertTrue(events.empty())


if __name__ == '__main__':
    unittest.main()
