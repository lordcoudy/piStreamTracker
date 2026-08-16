"""Optional camera record-API token."""

import unittest

from pistream.camera_auth import auth_headers, extract_bearer, token_ok


class TokenOkTests(unittest.TestCase):
    def test_empty_expected_allows_anyone(self):
        self.assertTrue(token_ok(None, None))
        self.assertTrue(token_ok(None, ''))
        self.assertTrue(token_ok('anything', ''))

    def test_mismatch_rejected(self):
        self.assertFalse(token_ok('wrong', 'secret'))
        self.assertFalse(token_ok(None, 'secret'))

    def test_match_accepted(self):
        self.assertTrue(token_ok('secret', 'secret'))


class BearerExtractTests(unittest.TestCase):
    def test_parses_authorization_header(self):
        self.assertEqual(extract_bearer('Bearer secret'), 'secret')
        self.assertIsNone(extract_bearer('Basic x'))
        self.assertIsNone(extract_bearer(None))

    def test_auth_headers_omitted_when_empty(self):
        self.assertEqual(auth_headers(None), {})
        self.assertEqual(auth_headers('tok'), {'Authorization': 'Bearer tok'})
