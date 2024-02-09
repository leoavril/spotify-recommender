import unittest
from app import app
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from util.helpers import parsePlaylistLink
import pandas
import os


class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ['SPOTIFY_CLIENT_ID'],
                                                                        client_secret=os.environ['SPOTIFY_CLIENT_SECRET']))

    def test_read_page(self):
        # check if the page is loaded
        self.assertTrue(self.app.testing)

    def test_predict_correct_playlist_link(self):
        # Test predicting correct play link
        playlist_link = r"https://open.spotify.com/playlist/37i9dQZF1DWXdiK4WAVRUW"
        response = self.app.post(
            '/predict', data=dict(playlist=playlist_link), follow_redirects=True)
        self.assertEqual(response.status_code, 200,
                         "Response should be 200 OK")

    def test_predict_wrong_playlist_link(self):
        # Test predicting wrong playlist link
        playlist_link = "https://wrong.link"
        response = self.app.post(
            '/predict', data=dict(playlist=playlist_link), follow_redirects=True)
        self.assertEqual(response.status_code, 400,
                         "Response should be 400")

    def test_function_returns_list_of_dict(self):
        playlist_link = r"https://open.spotify.com/playlist/37i9dQZF1DWXdiK4WAVRUW"
        parsedPlaylist = parsePlaylistLink(self.sp, playlist_link)
        self.assertEqual(type(parsedPlaylist), list)
        self.assertEqual(type(parsedPlaylist[0]), dict)
        self.assertEqual(type(parsedPlaylist[1]), pandas.core.series.Series)


if __name__ == '__main__':
    unittest.main()
