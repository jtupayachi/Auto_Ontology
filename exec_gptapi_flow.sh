#!/bin/bash

# Define the modes and their descriptions
MODES=("a" "GPT_Generate Ontology Files - No Properties" \

       "b" "GPT_Generate Ontology Files - Including Properties" 
       
       "c" "OLLAMA_Generate Ontology Files - - No Properties" 
    #    "mode3" "mode3 description"
       )

# Show menu and get result
CHOICE=$(whiptail --title "Ontology Creator" --menu "Choose a mode" 15 60 3 \
"${MODES[@]}" 3>&1 1>&2 2>&3)

exitstatus=$?

if [ $exitstatus = 0 ]; then
    echo "You chose $CHOICE. Exit status was $exitstatus."
    # Here you can handle what happens when a particular mode is selected
    case $CHOICE in
        a)
            # Handle mode1
            echo "Handling PDF" && source /home/jose/RECOIL_Auto_Onotology/gptapi_flow/.venv/bin/activate && python3 /home/jose/RECOIL_Auto_Onotology/gptapi_flow/onto_main.py 2>&1 | tee /home/jose/RECOIL_Auto_Onotology/logs/gptapi_flow_log.log
            ;;
        b)
            # Handle mode2
            echo "Handling PDF" && source /home/jose/RECOIL_Auto_Onotology/gptapi_flow/.venv/bin/activate && python3 /home/jose/RECOIL_Auto_Onotology/gptapi_flow/onto_main2.py 2>&1 | tee /home/jose/RECOIL_Auto_Onotology/logs/gptapi_flow_log2.log
            ;;
        c)
            # Handle mode1
            echo "Handling PDF" && source /home/jose/RECOIL_Auto_Onotology/gptapi_flow/.venv/bin/activate && python3 /home/jose/RECOIL_Auto_Onotology/gptapi_flow/onto_main3.py 2>&1 | tee /home/jose/RECOIL_Auto_Onotology/logs/gptapi_flow_log3.log
            ;;
        # mode3)
        #     # Handle mode3
        #     echo "Handling mode3"
        #     ;;
    esac
else
    echo "You chose to cancel. Exit status was $exitstatus."
fi