import os
import random
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask import Flask, session, redirect, url_for, request, render_template, Response, jsonify
import cv2
from deepface import DeepFace
import time
from collections import Counter
from spotipy.cache_handler import FlaskSessionCacheHandler

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(64)

client_id = 'f13c31f4ef944d45a90cff577b0af6a8'
client_secret = '58a684b699d347ebaa443e36f0db0017'
redirect_uri = 'http://localhost:8237/callback'
scope = 'playlist-modify-private playlist-modify-public playlist-read-private'

cache_handler = FlaskSessionCacheHandler(session)
sp_oauth = SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope=scope,
    cache_handler=cache_handler,
    show_dialog=True
)

sp = spotipy.Spotify(auth_manager=sp_oauth)

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the emotion detection function
def detect_emotion_from_video():
    cap = cv2.VideoCapture(0)  # Open the camera
    start_time = time.time()
    duration = 5  # Capture for 5 seconds
    emotion_counter = Counter()

    while True:
        success, frame = cap.read()
        if not success:
            print("Camera frame not read successfully")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            print("No face detected")
        else:
            for (x, y, w, h) in faces:
                face_roi = frame[y:y + h, x:x + w]
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                if result and 'dominant_emotion' in result[0]:
                    mood = result[0]['dominant_emotion']
                    emotion_counter[mood] += 1
                    print(f"Detected Emotion: {mood}")  # Debugging print

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, mood, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Stop after 5 seconds
        if time.time() - start_time > duration:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Return the most detected emotion
    if emotion_counter:
        most_common_emotion = emotion_counter.most_common(1)[0][0]
        print(f"Most Detected Emotion: {most_common_emotion}")  # Debugging print
        return most_common_emotion
    else:
        print("No emotions detected, defaulting to neutral")
        return "neutral"

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        cap = cv2.VideoCapture(0)  # Open the camera every time this function runs

        if not cap.isOpened():  # Check if the camera is accessible
            return "Error: Camera not accessible"

        # Setup face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            success, frame = cap.read()  # Capture a frame
            if not success:
                print("Camera frame not read successfully")
                break

            # Convert frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y + h, x:x + w]

                    # Perform emotion detection (optional, uncomment if using DeepFace)
                    try:
                        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                        emotion = result[0]['dominant_emotion']
                    except:
                        emotion = "Unknown"  # Default emotion if not detected

                    # Draw face bounding box and emotion text
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Encode frame to JPEG for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()  # Release the camera after the stream ends
        cv2.destroyAllWindows()  # Clean up

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_emotion')
def detect_emotion():
    detected_emotion = detect_emotion_from_video()
    print(f"Detected Emotion: {detected_emotion}")
    session['detected_emotion'] = detected_emotion
    return jsonify({'emotion': detected_emotion})

def get_songs_based_on_mood(mood):
    mood_keywords = {
        'angry': 'hard rock, aggressive rap, heavy metal, intense electronic',
        'disgust': 'dark ambient, industrial, experimental, eerie beats',
        'fear': 'chillwave, darkwave, experimental electronic, moody instrumental',
        'happy': 'upbeat pop, feel-good anthems, party tracks, dance music',
        'sad': 'acoustic ballads, mellow indie, somber R&B, sad pop',
        'surprise': 'high-energy dance, electronic bangers, pop remixes',
        'neutral': 'rap, hip-hop, chill beats, smooth jazz',
    }

    if mood not in mood_keywords:
        return []

    results = sp.search(q=mood_keywords[mood], type='track', limit=20)

    results = sp.search(q=mood_keywords[mood], type='track', limit=50)  # Increased limit to fetch more songs

    song_uris = []
    song_titles = set()  # To keep track of unique song titles

    for track in results['tracks']['items']:
        if track['name'] not in song_titles:
            song_uris.append(track['uri'])
            song_titles.add(track['name'])  # Add song title to the set

    return song_uris

@app.route('/')
def home():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)
    return redirect(url_for('create_playlist'))

@app.route('/callback')
def callback():
    sp_oauth.get_access_token(request.args['code'])
    return redirect(url_for('create_playlist'))

@app.route('/create_playlist', methods=['GET', 'POST'])
def create_playlist():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)

    return render_template('index.html')

@app.route('/generate_playlist', methods=['POST'])
def generate_playlist():
    detected_emotion = request.form.get('emotion', 'neutral')
    num_songs = int(request.form.get('num_songs', 10))

    user = sp.current_user()
    playlist = sp.user_playlist_create(user['id'], f"{detected_emotion.capitalize()} Vibes", public=False)

    song_uris = get_songs_based_on_mood(detected_emotion)

    num_songs = min(num_songs, len(song_uris))

    if not song_uris:
        return jsonify({"message": f"No songs found for mood '{detected_emotion}'."})

    sp.user_playlist_add_tracks(user['id'], playlist['id'], random.sample(song_uris, num_songs))

    return jsonify({"message": f"Playlist created!", "playlist_url": playlist['external_urls']['spotify']})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(port=8237, debug=True)
