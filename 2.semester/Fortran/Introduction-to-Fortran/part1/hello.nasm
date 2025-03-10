
	;; nasm syntax, more modern than the AT&T syntax.
	;; nasm -f elf64 -o hello.o hello.nasm
	;; ld -o hello hello.o
	;; ./hello


global _start
section .text
_start:
	mov rax, 1        ; write(
	mov rdi, 1        ;   STDOUT_FILENO,
	mov rsi, msg      ;   "Hello, world!\n",
	mov rdx, msglen   ;   sizeof("Hello, world!\n")
	syscall           ; );
	mov rax, 60       ; exit(
	mov rdi, 0        ;   EXIT_SUCCESS
	syscall 	          ; );

section .rodata
msg: db "Hello, world!", 10 	; 10 is newline.
msglen: equ $ - msg		; 14 chars in total.
