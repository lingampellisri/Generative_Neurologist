import { render, screen } from '@testing-library/react';
import OTP from './authentication';

test('renders learn react link', () => {
  render(<OTP />);
  const linkElement = screen.getByText(/learn react/i);
  expect(linkElement).toBeInTheDocument();
});