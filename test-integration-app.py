from main import SpotifyExplorer
from util.helpers import parsePlaylistLink
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import unittest
from app import app
import os


class IntegrationTestFlaskApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ['SPOTIFY_CLIENT_ID'],
                                                                        client_secret=os.environ['SPOTIFY_CLIENT_SECRET']))
        self.spotify_explorer = SpotifyExplorer(numFiles=0, retrainNNC=False)

    def test_read_page(self):
        # check if the page is loaded
        self.assertTrue(self.app.testing)

    def test_predict_playlist_link(self):
        # Test predicting correct play link
        playlist_link = r"https://open.spotify.com/playlist/37i9dQZF1DWXdiK4WAVRUW"
        response = self.app.post(
            '/predict', data=dict(playlist=playlist_link), follow_redirects=True)

        self.assertEqual(response.status_code, 200)

        parsedPlaylist = parsePlaylistLink(self.sp, playlist_link)
        playlist = parsedPlaylist[0]
        playlist_series = parsedPlaylist[1]

        tracks_embed = self.spotify_explorer.predictPlaylist(playlist_series)
        self.assertIsNotNone(tracks_embed)

    def test_html_contains_songs(self):
        playlist_link = r"https://open.spotify.com/playlist/37i9dQZF1DWXdiK4WAVRUW"
        response = self.app.post(
            '/predict', data=dict(playlist=playlist_link), follow_redirects=True)
        split_code = response.data.split(b"<li>")
        predicted_songs_tags = split_code[1:-1]

        self.assertGreater(len(predicted_songs_tags), 3)


if __name__ == '__main__':
    unittest.main()
