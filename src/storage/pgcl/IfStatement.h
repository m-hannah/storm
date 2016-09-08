/* 
 * File:   IfStatement.h
 * Author: Lukas Westhofen
 *
 * Created on 11. April 2015, 17:42
 */

#ifndef IFSTATEMENT_H
#define	IFSTATEMENT_H

#include "src/storage/pgcl/CompoundStatement.h"
#include "src/storage/pgcl/BooleanExpression.h"
#include "src/storage/pgcl/PgclProgram.h"

namespace storm {
    namespace pgcl {
        /**
         * This class represents if statements. Any if statement has a condition
         * which is saved as a boolean expression, and a statement body which is
         * again a PGCL program. Thus, an if statement is a compound statement.
         * It is possibly for if statements to have one else body, but not
         * mandatory.
         */
        class IfStatement : public CompoundStatement {
        public:
            IfStatement() = default;
            /**
             * Creates an if statement which saves only an if body.
             * @param condition The guard of the statement body.
             * @param body The if body.
             */
            IfStatement(storm::pgcl::BooleanExpression const& condition, std::shared_ptr<storm::pgcl::PgclBlock> const& body);
            /**
             * Creates an if statement with an if and an else body.
             * @param condition The guard of the if body.
             * @param ifBody The if body.
             * @param elseBody The else body.
             */
            IfStatement(storm::pgcl::BooleanExpression const& condition, std::shared_ptr<storm::pgcl::PgclBlock> const& ifBody, std::shared_ptr<storm::pgcl::PgclBlock> const& elseBody);
            IfStatement(const IfStatement& orig) = default;
            virtual ~IfStatement() = default;
            std::size_t getNumberOfOutgoingTransitions();
            void accept(class AbstractStatementVisitor&);
            /**
             * Returns the if body of the if statement.
             * @return The if body.
             */
            std::shared_ptr<storm::pgcl::PgclBlock> getIfBody();
            /**
             * Returns the else body of the if statement, if present. Otherwise
             * it throws an excpetion.
             * @return The else body.
             */
            std::shared_ptr<storm::pgcl::PgclBlock> getElseBody();
            /**
             * Returns true iff the if statement has an else body.
             */
            bool hasElse();
            /**
             * Returns the guard of the if statement.
             * @return The condition.
             */
            storm::pgcl::BooleanExpression& getCondition();
        private:
            /// The if body is again a PGCL program.
            std::shared_ptr<storm::pgcl::PgclBlock> ifBody;
            /// The else body is again a PGCL program.
            std::shared_ptr<storm::pgcl::PgclBlock> elseBody;
            /// Memorizes if an else body was set. Set to false by default.
            bool hasElseBody = false;
            /// Saves the guard of the if statement.
            storm::pgcl::BooleanExpression condition;
        };
    }
}

#endif	/* IFSTATEMENT_H */

