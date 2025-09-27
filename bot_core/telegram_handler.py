import httpx

class TelegramHandler:
    def __init__(self, config):
        self.token = config['token']
        self.chat_id = config['chat_id']
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    async def send_message(self, message):
        url = f"{self.base_url}/sendMessage"
        payload = {'chat_id': self.chat_id, 'text': message}
        try:
            async with httpx.AsyncClient() as client:
                await client.post(url, json=payload)
        except Exception as e:
            # Log this error but don't crash the bot
            print(f"Failed to send Telegram message: {e}")