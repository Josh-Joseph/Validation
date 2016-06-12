#lang racket

(require (for-syntax racket/syntax))

;;=====================================================================

(struct 
    Construct
  (tag pos-args kw-args)
  #:transparent)

;;=====================================================================

;;
;; Our Three worlds:
;;
;;    1) Math World : the most abstract world where constructs can be
;;    2) Computational World : A world with all idealized computation models.
;;          *every* construct in Computational-World is also in Math-World
;;          but the mapping is one to many and not everything in Math-World
;;          is in Computational-World.
;;          Algorithms live in Computational-World.
;;    3) Real World : Everything in Real-World is an approximation of something
;;          in Computational-World.  The links exist with given errors.
;;          Only Real-World construct can be processed and run in the real
;;          world (hence the name).

(struct
    MathWorld-Construct
  Construct
  ()
  #:transparent)

(struct
    ComputationalWorld-Construct
  Construct
  ()
  #:transparent)

(struct 
    RealWorld-Construct
  Construct
  ()
  #:transparent)

;;=====================================================================

;;
;; Maps a construct world identifier to a set of 
;; prefixes for names

(begin-for-syntax   
 
 ;; The internal mapping hatshtable
 (define %construct-world-identifier->name-preffixes
   (make-hasheq
    '( (math-world "mw:")
       (computation-world "cw:")
       (real-world "rw:"))))
 
 ;; Default prefixes
 (define %DEFAULT-NAME-PREFFIXES
   '(""))
 
 ;;
 ;; Function which maps from world identifier to prefixes
 (define (world->name-prefixes world-identifier)
   (cons
    (format "~a:" world-identifier)
    (hash-ref 
     %construct-world-identifier->name-preffixes
    world-identifier
    %DEFAULT-NAME-PREFFIXES)))

 )
 
;;=====================================================================

;; Mapping between a world identifier and the
;; base structure for constructs
;;
;; Map[ symbol => ( constructor, predicate ) ]
(define %construct-world-identifier->base-structure-info
  (make-hasheq
   (quasiquote 
    ( (math-world (unquote MathWorld-Construct) 
		  (unquote MathWorld-Construct?))
      (computation-world (unquote ComputationalWorld-Construct) 
			 (unquote ComputationalWorld-Construct?))
      (real-world (unquote RealWorld-Construct) 
		  (unquote RealWorld-Construct?)) ))))

;; The default Construct constructor
(define %DEFAULT-BASE-CONSTRUCT-INFO (list Construct Construct?) )

;; Function which return the base Construct info
;; for a given workd identifier
(define (world->base-construct-info world)
  (hash-ref 
   %construct-world-identifier->base-structure-info
   world
   %DEFAULT-BASE-CONSTRUCT-INFO))

;;=====================================================================


;; Syntax testing
(define-syntax (define/tcs stx)
  (syntax-case stx ()
    [ (define/tcs world-ident ident)
      (with-syntax ([normalized-ident 
		     (format-id #'world-ident "tcs:~a" #'world-ident)])
	#'(define normalized-ident 1))]))

;; syntax for creating a new construct
(define-syntax (define/construct stx)
  (syntax-case stx ()
    [ (define/construct world-ident construct-ident . config)
      (begin
	(unless (and (identifier? #'world-ident)
		     (identifier? #'construct-ident))
	  (raise-syntax-error 
	   #f 
	   "Need both world and construct to be identifiers"
	   #'world-ident))
	(with-syntax* ([tag (format-id #'world-ident 
				       "~a:~a"
				       (syntax-e #'world-ident)
				       (syntax-e #'construct-ident))]
		       [world-ident-val (format-symbol "~a" #'world-ident)]
		       [prefixes (world->name-prefixes (syntax-e #'world-ident))]
		       [maker-idents 
			(map (lambda (p) 
			       (format-id #'construct-ident 
					  "~a~a"
					  p #'construct-ident))
			     (syntax->datum #'prefixes))]
		       [maker-lambda-ident
			(format-id #'construct-ident
				   "construct:maker:~a:~a"
				   #'world-ident
				   #'construct-ident)]
		       [pred-idents
			(map (lambda (p)
			       (format-id #'construct-ident
					  "~a~a?"
					  p #'construct-ident))
			     (syntax->datum #'prefixes))]
		       [pred-lambda-ident
			(format-id #'construct-ident
				   "construct:predicate:~a:~a"
				   #'world-ident
				   #'construct-ident)]
		       [all-define-idents
			(for/fold ([defs '()])
			    ([maker (syntax-e #'maker-idents)]
			     [pred  (syntax-e #'pred-idents)])
			  (values (append defs (list maker pred))))]
		       [definitions
			 (for/fold ([defs '()]) 
			     ([ident-maker (syntax->datum #'maker-idents)]
			      [ident-pred (syntax->datum #'pred-idents)])
			   (values (append defs (list #'maker-lambda-ident #'pred-lambda-ident))))])
		      #'(define-values 
			  all-define-idents
			  (let* ([info 
				  (world->base-construct-info 'world-ident)]
				 [constructor (first info)]
				 [predicate (second info)]
				 [maker-lambda-ident
				  (make-keyword-procedure
				   (lambda (kws kw-args . rest)
				     (constructor 'tag 
						  rest 
						  (list kws kw-args))))]
				 [pred-lambda-ident
				  (lambda (x)
				    (and (predicate x)
					 (eq? (Construct-tag x)
					      'tag)))])
			    (values . definitions))))) ]))

				 
	      

;;=====================================================================
;;=====================================================================
;;=====================================================================
;;=====================================================================
;;=====================================================================
;;=====================================================================
;;=====================================================================
;;=====================================================================
