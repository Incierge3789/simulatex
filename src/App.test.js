import React from 'react';
import { render, screen } from '@testing-library/react';
import { act } from 'react';
import App from './App';

test('renders chat interface elements', () => {
  render(<App />);
  
  const inputElement = screen.getByPlaceholderText('Type your message');
  expect(inputElement).toBeInTheDocument();

  const buttonElement = screen.getByText('Send');
  expect(buttonElement).toBeInTheDocument();

  const responseElement = screen.getByText('Response: Waiting for response...');
  expect(responseElement).toBeInTheDocument();
});
