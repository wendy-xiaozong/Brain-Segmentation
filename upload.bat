dos2unix C:\Users\54344\PycharmProjects\MRBrainS18\Code\runs.sh
scp -r C:\Users\54344\PycharmProjects\MRBrainS18\Code\* jueqi@cedar.computecanada.ca:/home/jueqi/projects/def-jlevman/jueqi/MRBrainS18/Code
git init
git add -A C:\Users\54344\PycharmProjects\MRBrainS18\Code\*
git commit -m "message"
git push origin master -f