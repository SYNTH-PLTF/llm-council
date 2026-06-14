import { act, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { Typewriter } from './Typewriter'

describe('Typewriter', () => {
  it('streams the text to completion', () => {
    vi.useFakeTimers()
    try {
      render(<Typewriter text="hello world" speed={5} />)
      act(() => {
        vi.advanceTimersByTime(200)
      })
      expect(screen.getByTestId('typewriter').textContent).toBe('hello world')
    } finally {
      vi.useRealTimers()
    }
  })
})
