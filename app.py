# pythonspot.com
from flask import Flask, render_template, flash, request
from  MoodOfSong.code.FinalModel import ngram_vectorize, model_predict
import pickle
from googletrans import Translator
translator = Translator()
classifier = pickle.load(open('Finalmodel.pkl', 'rb'))


# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)

pickle_in= open('Finalmodel.pkl', 'rb')
Mymodel= pickle.load(pickle_in)




@app.route("/", methods=['GET', 'POST'])
def lyrics_form():
    if request.method == 'POST':
        lyrics = request.form.get('lyrics')
        translation = translator.translate(lyrics)
        lyricData=translation.text
        value, value2=model_predict(lyricData)
        probabvalue=" "
        if (value==['happy']):
            probavalue =(1- (value2[:, 1])) * 100

            val="Happy"


            return """
            <h1> Model predicted that mood of the song is {} </h1>
            <h2> The probability of song to be {} is {}%</h2>
            <img src="https://media.giphy.com/media/1MTLxzwvOnvmE/giphy.gif" style="max-width:30%;height:40%;">
            <h3>
                        Translated Song:
                         </h3>
                         <h4>{}. 
                        </h4>
                        
                    
    
                        """.format(val, val, probavalue, translation.text)


        else:
            probabvalue = ( (value2[:, 1])) * 100
            val = "Sad"
            print(value)
            probabvalue = (value2[:, 1]) * 100
            return """
            <h1> Mood of the song is {} </h1>
            <h2> The probability : {} is {}%</h2>
            <h3><img src="https://media1.tenor.com/images/30e8e3e8c85accdcd0702ef2bb859e84/tenor.gif?itemid=5823074" style="max-width:30%;height:40%;">
                        </h3>
                        <h3>Translated Song:
                         </h3>
                         <h4>{}. 
                        </h4>
            

                        """.format(val, val, probabvalue, translation.text)


    return '''
    <form method ="POST">
    <body background="https://images.unsplash.com/photo-1505740420928-5e560c06d30e?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1950&q=80" style="max-width:30%;height:40%;">
        <h1>Welcome to K3G Music</h1>
        <h2>Please enter the Lyrics below:</h2>
        <textarea name="lyrics" rows="30" cols="90" value="Enter ther lyrics" style="background-color:#FCF5D8;color:#AD8C08;"></textarea></br>
        <input type ="submit" value="Submit songtext" name="sbmtbtn"></form>
    '''

if __name__ == "__main__":
    # Getting the classifier ready
    app.run()

#<input type="text" name="lyrics" value="Enter the lyrics" cols="30" rows="10">
# <body>
    #   <form method ="POST">
    #     <h1>Please enter the songtext below:</h1>
    #     <textarea name="lyrics" rows="10" cols="30" value="Enter ther lyrics"></textarea></br>
    #     <input type ="submit" value="Submit songtext" name="sbmtbtn"></form>
    # </body>