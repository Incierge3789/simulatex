import React, { useState, useCallback } from 'react';

function ChatComponent() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setResponse('');
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
      });

      if (!res || !res.ok) {
        throw new Error(res ? `HTTP error! status: ${res.status}` : 'Network response was not ok');
      }

      const data = await res.json();
      setResponse(data.response || 'No response from API');
    } catch (error) {
      console.error('There was a problem with the fetch operation:', error);
      setResponse(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [message]);

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input 
          type="text" 
          value={message} 
          onChange={(e) => setMessage(e.target.value)} 
          placeholder="Type your message"
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
      <p>Response: {response || 'Waiting for response...'}</p>
    </div>
  );
}

export default ChatComponent;
