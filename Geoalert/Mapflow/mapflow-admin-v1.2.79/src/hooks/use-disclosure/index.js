import { useState, useCallback } from "react";

export function useDisclosure(initialOpened = false) {
  const [isOpen, setIsOpen] = useState(initialOpened);
  const onClose = useCallback(() => setIsOpen(false), []);
  const onOpen = useCallback(() => setIsOpen(true), []);
  return { isOpen, onClose, onOpen };
}
