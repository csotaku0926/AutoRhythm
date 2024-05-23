// note peroidically falls down
const SPEED = 1;
var stop_time = 2400; 
const INTERVAL = 50;
const JUDGE_HEIGHT = 80;
// available keys
const keys = ['a','s','d','f','h','j','k','l'];
var keys_dict = {};
let i = 0;
for(let k of keys)
{
    keys_dict[k] = i;
    i += 1;
}
    
var timer = 0;

// notes element: div
var notes_a = []; 
var notes_s = [];
var notes_d = []; 
var notes_f = [];
var notes_h = []; 
var notes_j = [];
var notes_k = []; 
var notes_l = [];

// second element
var second_a = []; 
var second_s = [];
var second_d = []; 
var second_f = [];
var second_h = []; 
var second_j = [];
var second_k = []; 
var second_l = [];

// each track
var track_a = document.getElementById("track_a");
var track_s = document.getElementById("track_s");
var track_d = document.getElementById("track_d");
var track_f = document.getElementById("track_f");
var track_h = document.getElementById("track_h");
var track_j = document.getElementById("track_j");
var track_k = document.getElementById("track_k");
var track_l = document.getElementById("track_l");

// combine 
var all_notes = [notes_a, notes_s, notes_d, notes_f, notes_h, notes_j, notes_k, notes_l];
var all_second = [second_a, second_s, second_d, second_f, second_h, second_j, second_k, second_l];
var tracks = [track_a, track_s, track_d, track_f, track_h, track_j, track_k, track_l];

// combo variables
let combo_num_span = document.getElementById("combo_num");
let combo_num = 0;

// score for hitting notes
var hit_score = 0.0;

// want to make total 1000000
const total_score = 1000000;
var perfect_score = 140;
var good_score = 0.5 * perfect_score;

// every 10 combo have 0.1 addition
var combo_multiply = 1 + parseInt(combo_num / 10) * 0.1;
// the closet note of each track to detection line
var first_note = [0, 0, 0, 0, 0, 0, 0, 0];

function update_score(score){
    let hit_score = document.getElementById("hit_score");
    let str_score = String(score);
    let return_score = '';

    // zero padding
    for(let i = 0; i < 7 - str_score.length; i++)
        return_score += '0';
    return_score += str_score;
    hit_score.innerText = return_score;
};

function draw_note (json) {
    // read note from json file
    for(let note of json["notes"])
    {
        let key = keys_dict[note["track"]];
        (all_second[key]).push(note["second"]);
    }
    stop_time = json["stop_time"];
    // every `INTERVAL` does
    var timeID = window.setInterval(() => {
        timer += SPEED;
        
        // check each track
        for(let i = 0; i < keys.length; i += 1)
        {
            let this_first_note = first_note[i];
            let this_second = all_second[i];
            // reach time
            if(this_first_note < this_second.length && this_second[this_first_note] == timer)
            {
                let this_track = tracks[i];
                // add note to track
                let newNote = document.createElement("div");
                newNote.classList.add("note");
                newNote.style.top = '0%';
                this_track.appendChild(newNote);
                all_notes[i].push(newNote);
                first_note[i] += 1;
            }

        }          
        // every note drop down by time
        let i = 0;

        for(let n = 0; n < all_notes.length; n++)
            for (i = 0; i < all_notes[n].length; i += 1)
            {
                let notes = all_notes[n];
                let seconds = all_second[n];
                let note = notes[i];
                let top_int = timer - seconds[i];
                note.style.top = `${top_int}%` ;
                // reach judge line
                if(top_int > JUDGE_HEIGHT)
                {
                    // miss
                    // update combo
                    combo_num = 0;
                    combo_num_span.innerText = `${combo_num}`;
                    combo_num_span.style.color = "white";

                    let text = document.getElementById(`score_${keys[n]}`);
                    // add miss text
                    text.innerText = "MISS";
                    text.classList.add("miss");

                    // clear effect after a period
                    window.setTimeout(()=>{
                        text.classList.remove("miss");
                    }, 500 / SPEED);
                    // remove the poped note 
                    (notes.splice(i,1)[0]).remove();
                    seconds.splice(i,1);
                    i-=1;
                    first_note[n] -=1;
                }  
            }

        // stop timer
        if(timer >= stop_time)
        {
            console.log("End playing");
            window.clearInterval(timeID);
            if (audio) {
                audio.pause();
                audio.currentTime = 0;
            }
        }
    }, INTERVAL);      
};

// add the button interaction to play audio
let confirm_button = document.getElementById("confirm_button");
confirm_button.addEventListener('click',(() => {
    console.log("clicked!");
    confirm_button.style.display = "none";

    // play audio after first note is ready
    let audio = document.getElementById("selected_song");
    
    setInterval(() => {audio.play();}, INTERVAL * JUDGE_HEIGHT);

    /**
     * Since most of web servers have blocked direct file system access
     * (i.e. cannot use file protocol load on http website)
     * 
     * we must run this code on a server
     * otherwise there will be CORS error
     * 
     */

    fetch ("beat_map/percussion.json")
    .then (res => res.json())
    .then ((json) => draw_note(json))
    
}));

var isPressed = {};
for(let k of keys)
    isPressed[k] = false;

// light up div and judge when pressed the keyboard
window.addEventListener("keydown", (e) => {
    if (!keys.includes(e.key)) return;
    
    let jl = document.getElementById("jl_" + e.key);
    let bt = document.getElementById("bt_" + e.key);
    
    // if not pressed yet
    if(!jl.classList.contains("judge_line_pressed"))
        jl.classList.add("judge_line_pressed");
    bt.style.color = 'white';

    // detect perfect, good, bad or miss
    // detect on the closet note
    let this_key = keys_dict[e.key];
    let this_notes = all_notes[this_key];
    let this_second = all_second[this_key];
    let top_int = timer - this_second[0];
    
    if(!isPressed[e.key])
    {
        let text = document.getElementById(`score_${e.key}`);
        // judged good
        if(top_int >= 75 && top_int <= 77)
        {
            jl.classList.add("judge_line_good");
            // add good text
            text.innerText = "GOOD";
            text.classList.add("good");
            // add score
            hit_score += good_score * combo_multiply;
            update_score(hit_score);
            // update combo
            combo_num += 1;
            combo_num_span.innerText = `${combo_num}`; 
            if(combo_num_span.style.color == "gold")
                combo_num_span.style.color = "rgb(8, 142, 225)";
            // clear effect after a second
            window.setTimeout(()=>{
                text.classList.remove("good");
                jl.classList.remove("judge_line_good");
            }, 500 / SPEED);
            // remove note
            (this_notes.splice(0,1)[0]).remove();
            this_second.splice(0,1);
            first_note[this_key] -=1;
        }
        // perfect
        else if(top_int >= 78 && top_int <= 80)
        {
            // judge line
            jl.classList.add("judge_line_perfect");
            // add perfect text
            text.innerText = "PERFECT";
            text.classList.add("perfect");
            // add score
            hit_score += perfect_score * combo_multiply;
            update_score(hit_score);
            // update combo
            combo_num += 1;
            combo_num_span.innerText = `${combo_num}`; 
            // clear effect after a second
            window.setTimeout(()=>{
                text.classList.remove("perfect");
                jl.classList.remove("judge_line_perfect");
            }, 500 / SPEED);
            // remove note
            (this_notes.splice(0,1)[0]).remove();
            this_second.splice(0,1);
            first_note[this_key] -=1;
        }
    }
    
});

// detect if pressed
window.addEventListener("keypress", (e) =>{
    isPressed[e.key] = true;
});

// back off when release keyboard
window.addEventListener("keyup", (e) => {
    if(keys.includes(e.key))
    {
        let jl = document.getElementById("jl_" + e.key);
        let bt = document.getElementById("bt_" + e.key);
        // release key
        isPressed[e.key] = false;
        
        // if pressed
        if(jl.classList.contains("judge_line_pressed"))
            jl.classList.remove("judge_line_pressed");
        bt.style.color = 'aquamarine'; 
    }
});

