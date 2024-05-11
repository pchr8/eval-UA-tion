git add . 
git commit -m "#22 Auto-push on deploying to serhii.net"
git push
rsync -ru --progress * redacted@redacted:/home/public/F/MA/presentation
